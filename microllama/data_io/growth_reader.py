"""Checksummed, append-only catalogs and non-repeating token-weighted sampling."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .blocks import DataWaitStopped
from .download import sha256
from .transfer import atomic_json

MODE = 'mixture-growth-v1'

class DataUnavailable(RuntimeError):
    pass

class GrowthExhausted(RuntimeError):
    pass

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load_catalog(directory, allow_failed=False, expected_sha256=None):
    directory = Path(directory)
    pointer = json.loads((directory / 'manifest.json').read_text())
    if expected_sha256 is not None and pointer['sha256'] != expected_sha256:
        raise ValueError('Parent catalog snapshot checksum mismatch')
    name = pointer['snapshot']
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('Unsafe catalog snapshot name')
    path = directory / 'snapshots' / name
    if sha256(path) != pointer['sha256']:
        raise ValueError('Catalog snapshot checksum mismatch')
    catalog = json.loads(path.read_text())
    if catalog['mode'] != MODE or catalog['schema_version'] != 1:
        raise ValueError('Unsupported growth catalog')
    if digest(catalog['recipe']) != catalog['recipe_id']:
        raise ValueError('Growth recipe identity mismatch')
    if catalog['status'] == 'failed' and not allow_failed:
        raise RuntimeError(f"Growth preparation failed: {catalog.get('error')}")
    names = set(catalog['sources'])
    if names != set(catalog['recipe']['sources']):
        raise ValueError('Catalog source set mismatch')
    weights = [v['weight'] for v in catalog['recipe']['sources'].values()]
    if any(not 0 < w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-12:
        raise ValueError('Invalid token mixture weights')
    return catalog

def publish_catalog(directory, catalog):
    directory = Path(directory)
    snapshots = directory / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    previous = None
    if (directory / 'manifest.json').exists():
        previous = json.loads((directory / 'manifest.json').read_text())
        old = load_catalog(directory, allow_failed=True)
        if old['recipe_id'] != catalog['recipe_id']:
            raise ValueError('Cannot change a published growth recipe')
        for source, entry in old['sources'].items():
            if entry['directory'] != catalog['sources'][source]['directory']:
                raise ValueError('Cannot change published source directory')
            for name, chunk in entry['chunks'].items():
                if catalog['sources'][source]['chunks'].get(name) != chunk:
                    raise ValueError('Cannot remove/change a published growth chunk')
    catalog = {**catalog, 'parent_snapshot_sha256': previous['sha256'] if previous else None,
               'generation': previous['generation']+1 if previous else 0}
    name = f"{catalog['generation']:08d}-{digest(catalog)[:16]}.json"
    atomic_json(snapshots / name, catalog)
    atomic_json(directory / 'manifest.json', {'mode': MODE, 'snapshot': name,
                'sha256': sha256(snapshots / name), 'generation': catalog['generation']})
    return catalog

def usable_targets(path, length, eos=7, pad=0):
    values = np.memmap(path, dtype='<u2', mode='r') if Path(path).stat().st_size else np.array([],dtype='<u2')
    n = max(0, (len(values)-1)//length)*length
    total = 0
    for start in range(0, n, 8_388_608):
        end = min(n, start+8_388_608)
        x, y = values[start:end], values[start+1:end+1]
        total += int(((x != eos) & (x != pad) & (y != pad)).sum())
    return total

class MixtureTokenBlocks:
    """One global owner; source cursors and deficit accounting survive restart.

    Completed sources never cycle. Catalog append is explicit and checksummed;
    an unavailable source returns control to the distributed polling loop.
    """
    def __init__(self, directory, sequence_length, seed=42):
        self.directory = Path(directory)
        self.sequence_length, self.seed = sequence_length, seed
        self.catalog = load_catalog(directory)
        self.recipe_id = self.catalog['recipe_id']
        recipe = self.catalog['recipe']
        if recipe['sequence_length'] != sequence_length or recipe['seed'] != seed:
            raise ValueError('Growth sequence/seed mismatch')
        self.eos, self.pad = recipe['eos'], recipe['pad']
        self.weights = {k:v['weight'] for k,v in recipe['sources'].items()}
        self.cursors = {k:{'current':None, 'cursor':0, 'consumed':[], 'known':{}} for k in self.weights}
        self.targets = {k:0 for k in self.weights}
        self.policy_origin = None
        self.mixture_transition = None
        self.retired_reader = None
        self.parent_tokens = None
        self.maps = {}
        self.orders = {}
        self._refresh()

    @property
    def known(self):
        return {f'{s}/{n}':v for s,c in self.cursors.items() for n,v in c['known'].items()}

    def _refresh(self):
        catalog = load_catalog(self.directory)
        if catalog['recipe_id'] != self.recipe_id:
            raise ValueError('Growth recipe changed')
        for source, cursor in self.cursors.items():
            chunks = catalog['sources'][source]['chunks']
            for name, old_hash in cursor['known'].items():
                if name not in chunks or chunks[name]['manifest_sha256'] != old_hash:
                    raise ValueError('Required growth chunk removed or changed')
            cursor['known'].update({k:v['manifest_sha256'] for k,v in chunks.items()})
        self.catalog = catalog

    def _open(self, source, name):
        if Path(name).name != name or name.endswith('.partial'):
            raise ValueError('Unsafe growth chunk')
        entry = self.catalog['sources'][source]
        directory = Path(entry['directory']) / name
        manifest = directory / 'manifest.json'
        if sha256(manifest) != self.cursors[source]['known'][name]:
            raise ValueError('Growth chunk manifest changed')
        meta = json.loads(manifest.read_text())
        pin = self.catalog['recipe']['sources'][source]
        if meta['source_revision'] != pin['revision'] or meta['tokenizer_sha256'] != self.catalog['recipe']['tokenizer_sha256']:
            raise ValueError('Growth chunk provenance mismatch')
        for split in ('train','validation'):
            file = directory / f'{split}.bin'
            if file.stat().st_size != meta['splits'][split]['tokens']*2 or sha256(file) != meta['splits'][split]['sha256']:
                raise ValueError('Corrupt growth token payload')
        if usable_targets(directory/'train.bin',self.sequence_length,self.eos,self.pad) != entry['chunks'][name]['valid_targets']:
            raise ValueError('Growth valid-target inventory mismatch')
        values = np.memmap(directory/'train.bin',mode='r',dtype='<u2') if meta['splits']['train']['tokens'] else np.array([],dtype='<u2')
        count = max(0,(len(values)-1)//self.sequence_length)
        key = int.from_bytes(hashlib.blake2b(f'{self.seed}:{source}:{name}'.encode(),digest_size=8).digest(),'little')
        self.maps[source] = values
        self.orders[source] = np.random.default_rng(key).permutation(count)

    def _row(self, source, *, consume=True):
        cursor = self.cursors[source]
        while True:
            name = cursor['current']
            if name is not None:
                if source not in self.maps:
                    self._open(source,name)
                if cursor['cursor'] < len(self.orders[source]):
                    start = int(self.orders[source][cursor['cursor']])*self.sequence_length
                    row = np.asarray(self.maps[source][start:start+self.sequence_length+1],dtype=np.int64)
                    valid = int(((row[:-1]!=self.eos)&(row[:-1]!=self.pad)&(row[1:]!=self.pad)).sum())
                    if valid:
                        if consume:
                            cursor['cursor'] += 1
                            self.targets[source] += valid
                        return row
                    cursor['cursor'] += 1
                    continue
                cursor['consumed'].append(name)
                cursor.update(current=None,cursor=0)
                self.maps.pop(source,None); self.orders.pop(source,None)
            candidates = sorted(set(cursor['known'])-set(cursor['consumed']))
            if not candidates:
                if self.catalog['complete']:
                    raise GrowthExhausted(f'No unconsumed {source} targets; repetition is prohibited')
                raise DataUnavailable(source)
            cursor['current'], cursor['cursor'] = candidates[0],0

    def state_dict(self):
        state = {'mode':MODE,'recipe_id':self.recipe_id,'sequence_length':self.sequence_length,
                'seed':self.seed,'parent_tokens':self.parent_tokens,'retired_reader':self.retired_reader,
                'source_targets':dict(self.targets),'sources':json.loads(json.dumps(self.cursors))}
        if self.policy_origin is not None:
            state['policy_origin'] = dict(self.policy_origin)
            state['mixture_transition'] = json.loads(json.dumps(self.mixture_transition))
        return state

    def load_state_dict(self,state):
        for key,value in (('mode',MODE),('recipe_id',self.recipe_id),('sequence_length',self.sequence_length),('seed',self.seed)):
            if state.get(key) != value:
                raise ValueError(f'Growth resume mismatch: {key}')
        self._refresh()
        if set(state['sources'])!=set(self.weights):
            raise ValueError('Growth checkpoint source set mismatch')
        targets = state['source_targets']
        if set(targets)!=set(self.weights) or any(type(v)is not int or v<0 for v in targets.values()):
            raise ValueError('Invalid source target counters')
        origin = state.get('policy_origin')
        if 'policy_origin' in state and not isinstance(origin,dict):
            raise ValueError('Invalid mixture policy origin')
        if origin is not None and (set(origin)!=set(self.weights) or any(
                type(v)is not int or not 0<=v<=targets[s] for s,v in origin.items())):
            raise ValueError('Invalid mixture policy origin')
        if state['parent_tokens'] is not None and (type(state['parent_tokens']) is not int or state['parent_tokens']<0):
            raise ValueError('Invalid historical token counter')
        transition = state.get('mixture_transition')
        if origin is None and 'mixture_transition' in state:
            raise ValueError('Mixture transition metadata requires a policy origin')
        if origin is not None:
            if not isinstance(transition,dict) or any(
                    not isinstance(transition.get(k),str) or len(transition[k])!=64 or any(
                        c not in '0123456789abcdef' for c in transition[k])
                    for k in ('parent_recipe_id','parent_catalog_sha256')):
                raise ValueError('Invalid mixture transition binding')
            if state['parent_tokens'] is None or type(transition.get('starting_tokens')) is not int or (
                    transition['starting_tokens']!=state['parent_tokens']+sum(origin.values())):
                raise ValueError('Mixture transition counters do not reconcile')
        for source,cursor in state['sources'].items():
            if source not in self.cursors:
                raise ValueError('Growth source set changed')
            for name,hash_ in cursor['known'].items():
                if self.cursors[source]['known'].get(name) != hash_:
                    raise ValueError('Checkpoint growth inventory changed')
            if len(set(cursor['consumed'])) != len(cursor['consumed']) or not set(cursor['consumed']).issubset(cursor['known']):
                raise ValueError('Invalid consumed growth chunks')
            if cursor['current'] is not None and (cursor['current'] not in cursor['known'] or cursor['current'] in cursor['consumed']):
                raise ValueError('Invalid current growth chunk')
            if type(cursor['cursor']) is not int or cursor['cursor']<0 or (cursor['current'] is None and cursor['cursor']!=0):
                raise ValueError('Invalid growth cursor')
        previous_maps, previous_orders = self.maps, self.orders
        self.maps, self.orders = {}, {}
        try:
            for source,cursor in state['sources'].items():
                if cursor['current']:
                    self._open(source,cursor['current'])
                    if cursor['cursor']>len(self.orders[source]):
                        raise ValueError('Invalid growth cursor')
        except BaseException:
            self.maps, self.orders = previous_maps, previous_orders
            raise
        self.cursors = json.loads(json.dumps(state['sources']))
        self.targets = dict(targets)
        self.policy_origin = dict(origin) if origin is not None else None
        self.mixture_transition = json.loads(json.dumps(state.get('mixture_transition')))
        self.parent_tokens, self.retired_reader = state['parent_tokens'],state['retired_reader']
        self._refresh()

    def migrate(self,old_state,parent_tokens):
        legacy = self.catalog['recipe']['legacy']
        path = Path(legacy['inventory'])
        if sha256(path) != legacy['inventory_sha256']:
            raise ValueError('Legacy inventory receipt changed')
        inventory = json.loads(path.read_text())
        expected = {k:v['manifest_sha256'] for k,v in inventory['chunks'].items()}
        if old_state.get('mode')!='growing' or old_state.get('known')!=expected:
            raise ValueError('Parent data cursor does not bind the verified starting sample')
        if old_state['sequence_length']!=self.sequence_length or old_state['seed']!=self.seed:
            raise ValueError('Parent cursor sequence/seed mismatch')
        if parent_tokens < self.catalog['initial_tokens']:
            raise ValueError('Initial stage is not complete')
        self.parent_tokens = parent_tokens
        self.retired_reader = json.loads(json.dumps(old_state))

    def migrate_mixture(self,state,parent_catalog,*,parent_catalog_sha256):
        """Explicit additive-source handoff bound to an immutable parent catalog.

        Existing source inventories and cursors stay intact. Only future targets
        follow the new weights; cumulative counters retain historical exposure.
        Exact resume continues to reject recipe/source changes.
        """
        if not isinstance(parent_catalog_sha256,str) or len(parent_catalog_sha256)!=64 or any(
                c not in '0123456789abcdef' for c in parent_catalog_sha256):
            raise ValueError('Mixture migration requires a bound parent catalog SHA-256')
        if self.parent_tokens is not None or any(self.targets.values()) or any(
                c['current'] is not None or c['cursor'] or c['consumed'] for c in self.cursors.values()):
            raise ValueError('Mixture migration requires an unused destination reader')
        parent = load_catalog(parent_catalog, expected_sha256=parent_catalog_sha256)
        if state.get('mode')!=MODE or state.get('recipe_id')!=parent['recipe_id']:
            raise ValueError('Mixture state does not bind the parent recipe')
        old_names = set(parent['sources'])
        if not old_names < set(self.weights) or set(state['sources'])!=old_names:
            raise ValueError('Mixture migration requires additive sources')
        old_controls = {k:v for k,v in parent['recipe'].items() if k!='sources'}
        new_controls = {k:v for k,v in self.catalog['recipe'].items() if k!='sources'}
        if old_controls!=new_controls:
            raise ValueError('Mixture migration changed recipe controls')
        if any(parent[k]!=self.catalog[k] for k in ('initial_tokens','target_tokens')):
            raise ValueError('Mixture migration changed cumulative acquisition budget')
        for source in old_names:
            old_pin = {k:v for k,v in parent['recipe']['sources'][source].items() if k!='weight'}
            new_pin = {k:v for k,v in self.catalog['recipe']['sources'][source].items() if k!='weight'}
            if old_pin!=new_pin or parent['sources'][source]!=self.catalog['sources'][source]:
                raise ValueError('Mixture migration changed existing source pins or inventory')
            expected = {k:v['manifest_sha256'] for k,v in parent['sources'][source]['chunks'].items()}
            if state['sources'][source]['known']!=expected:
                raise ValueError('Mixture state inventory differs from bound parent catalog')
        # Validate the old cursor/counters against the parent before constructing
        # a candidate. Neither rejection nor a corrupt payload mutates this reader.
        old = MixtureTokenBlocks(parent_catalog,self.sequence_length,self.seed)
        old.load_state_dict(state)
        if old.parent_tokens is None or old.parent_tokens+sum(old.targets.values())>=self.catalog['target_tokens']:
            raise ValueError('Mixture migration has no remaining cumulative budget')
        candidate = json.loads(json.dumps(state))
        candidate['recipe_id'] = self.recipe_id
        for source in set(self.weights)-old_names:
            candidate['source_targets'][source] = 0
            candidate['sources'][source] = json.loads(json.dumps(self.cursors[source]))
        candidate['policy_origin'] = dict(candidate['source_targets'])
        candidate['mixture_transition'] = {'parent_recipe_id':parent['recipe_id'],
            'parent_catalog_sha256':parent_catalog_sha256,
            'starting_tokens':old.parent_tokens+sum(old.targets.values())}
        staged = MixtureTokenBlocks(self.directory,self.sequence_length,self.seed)
        staged.load_state_dict(candidate)
        # Structural coverage and consumed counters must reconcile before commit;
        # incomplete acquisition may still wait for explicitly admitted data.
        staged.coverage(self.catalog['target_tokens'])
        for name in ('catalog','cursors','targets','policy_origin','mixture_transition',
                     'parent_tokens','retired_reader','maps','orders'):
            setattr(self,name,getattr(staged,name))

    def _balanced_source(self, counts, valid):
        """Choose using actual boundary-masked targets, without consuming lookahead.

        More than two sources can overshoot a one-row discrepancy bound when
        only the current largest deficit is considered. Compare the resulting
        deficits for every candidate; source cursors remain unchanged until the
        selected row is committed. Historical counters and policy origin stay
        intact across resume.
        """
        total = sum(counts.values())
        def score(source):
            n = valid[source]
            errors = [counts[s] + (n if s == source else 0) - (total+n)*w
                      for s, w in self.weights.items()]
            return max(map(abs, errors)), sum(e*e for e in errors), -self.weights[source], source
        return min(valid, key=score)

    def try_batch(self,size):
        self._refresh()
        before = self.state_dict()
        rows = []
        try:
            for _ in range(size):
                counts = self.targets if self.policy_origin is None else {
                    s:self.targets[s]-self.policy_origin[s] for s in self.weights}
                total = sum(counts.values())
                if len(self.weights) <= 2:
                    # Preserve the historical two-source sequence exactly.
                    source = max(self.weights,key=lambda k:(total*self.weights[k]-counts[k],self.weights[k],k))
                else:
                    candidates, missing = {}, []
                    for s in self.weights:
                        try:
                            candidates[s] = self._row(s, consume=False)
                        except (DataUnavailable, GrowthExhausted) as error:
                            missing.append(error)
                    if not candidates:
                        raise missing[0]
                    valid = {s: int(((row[:-1] != self.eos) & (row[:-1] != self.pad)
                                    & (row[1:] != self.pad)).sum())
                             for s, row in candidates.items()}
                    source = self._balanced_source(counts, valid)
                    projected = [counts[s] + (valid[source] if s == source else 0)
                                 - (total+valid[source])*w for s, w in self.weights.items()]
                    if max(map(abs, projected)) > self.sequence_length:
                        if missing:
                            raise missing[0]
                        raise ValueError('No next source row preserves the mixture bound')
                rows.append(self._row(source))
        except DataUnavailable:
            self.load_state_dict(before)
            return None
        except BaseException:
            self.load_state_dict(before)
            raise
        return np.stack(rows)

    def next_batch(self,size,stop_requested=None):
        rows = self.try_batch(size)
        if rows is None:
            raise DataWaitStopped('More admitted growth data is needed')
        return rows

    def coverage(self,target_tokens,minimum_ready=0):
        historical = self.parent_tokens if self.parent_tokens is not None else self.catalog['initial_tokens']
        required = target_tokens-historical-sum(self.targets.values())
        if required < 0 or target_tokens>self.catalog['target_tokens']:
            raise ValueError('Requested training target is outside the declared acquisition budget')
        totals={s:sum(v['valid_targets'] for v in e['chunks'].values()) for s,e in self.catalog['sources'].items()}
        remaining={s:totals[s]-self.targets[s] for s in totals}
        if any(v<0 for v in remaining.values()):
            raise ValueError('Consumed source counts exceed verified coverage')
        counts = self.targets if self.policy_origin is None else {
            s:self.targets[s]-self.policy_origin[s] for s in self.weights}
        elapsed = sum(counts.values())
        ready=max(0,min((remaining[s]+counts[s])/self.weights[s] for s in remaining)-elapsed)
        if ready<minimum_ready:
            raise ValueError(f'Insufficient balanced unconsumed data: {int(ready):,} targets ready')
        outstanding = {s:(elapsed+required)*self.weights[s]-counts[s] for s in self.weights}
        if self.catalog['complete'] and required>0 and any(
                remaining[s]==0 and outstanding[s]>0 for s in remaining):
            raise ValueError('Completed acquisition has an exhausted required source')
        if self.catalog['complete'] and ready+self.sequence_length < required:
            raise ValueError('Completed acquisition cannot cover remaining growth budget')
        return {'mode':MODE,'source_coverage':totals,'source_consumed':dict(self.targets),
                'balanced_unconsumed_targets':int(ready),'target_tokens':target_tokens,
                'remaining_required_targets':required,'parent_tokens':self.parent_tokens,
                'replay_targets':0,'acquisition_complete':self.catalog['complete']}
