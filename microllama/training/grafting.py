"""Offline teacher features and causal exact-suffix lookup for a bounded pilot.

Implements the core construction/retrieval idea in arXiv:2605.20948 (May 2026):
frequent recipient-side 2/3/4-grams index frozen teacher hidden states; the
longest matching suffix wins. Teacher tokenization happens only offline.

Our warm-start pilot adds a gated correction to the existing frozen Engram
backbone. Misses reproduce the parent; native Engram remains active on hits too.
The reader includes short causal convolution with explicit document resets.
This is an additive adaptation experiment, not the paper's from-scratch MoE
training with exclusive Engram fallback. No teacher is
loaded during recipient training or inference. xmem_pilot owns GPU admission,
budgets, evaluation and reader-only restart publication.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import hashlib
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from ..data_io.download import sha256
from ..data_io.transfer import atomic_json
from .xmem import TargetReader


ORDERS = (2, 3, 4)
ROW_BUDGETS = (32768, 16384, 16384)
MAX_SCAN_TOKENS = 64 * 65536


def control_ids(tokenizer_path: Path, extra=(0, 1, 7)) -> set[int]:
    document = json.loads(tokenizer_path.read_text())
    return set(extra) | {int(t['id']) for t in document.get('added_tokens', [])
        if t.get('special') or (t['content'].startswith('<|') and t['content'].endswith('|>'))}


def packed_windows(ids, order):
    """Collision-free uint64 keys for at most four uint16 IDs (order is separate)."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids > 65535)):
        raise ValueError('Expected one uint16-compatible token sequence')
    if order not in ORDERS:
        raise ValueError('Only 2/3/4-grams are supported')
    size = max(0, len(ids) - order + 1)
    result = np.zeros(size, dtype=np.uint64)
    for offset in range(order):
        result = (result << np.uint64(16)) | ids[offset:offset+size].astype(np.uint64)
    return result


def count_ngrams(ids, protected, counters):
    """Count only in-document windows; never join separate sampled slices."""
    ids = np.asarray(ids)
    blocked = np.isin(ids, list(protected))
    for order in ORDERS:
        packed = packed_windows(ids, order)
        valid = np.ones(len(packed), dtype=bool)
        for offset in range(order):
            valid &= ~blocked[offset:offset+len(packed)]
        keys, counts = np.unique(packed[valid], return_counts=True)
        counters[order].update({int(k): int(n) for k, n in zip(keys, counts)})


def unpack_key(key, order):
    return tuple((int(key) >> (16 * offset)) & 65535 for offset in reversed(range(order)))


def select_keys(counters, student, teacher, teacher_protected, budgets=ROW_BUDGETS, teacher_prefix=(1,)):
    """Rank on training frequency alone; preserve recipient keys, retokenize text."""
    selected = []
    for order, quota in zip(ORDERS, budgets):
        ranked = heapq.nsmallest(quota * 4, ((-count, key) for key, count in counters[order].items() if count >= 2))
        accepted = 0
        for negative_count, key in ranked:
            ids = unpack_key(key, order)
            text = student.decode(list(ids), skip_special_tokens=False)
            if '\ufffd' in text or not text.strip():
                continue  # Partial ByteLevel fragments are not trustworthy surface text.
            teacher_ids = teacher.encode(text, add_special_tokens=False).ids
            if not teacher_ids or len(teacher_ids) > 24 or set(teacher_ids) & teacher_protected:
                continue
            selected.append((ids, -negative_count, list(teacher_prefix) + teacher_ids))
            accepted += 1
            if accepted == quota:
                break
        if accepted == 0:
            raise ValueError(f'No usable repeated {order}-grams in the training sample')
    return selected


class ExactSuffixAddressing:
    """Exact keys include the current input token, never its next-token target.

    Zero denotes a miss. Separate dictionaries for each order avoid ambiguity;
    uint16 packing is reversible, not a lossy hash. Controls cannot enter keys.
    Chunk starts have no implicit past; cached decoding is intentionally unused.
    """
    def __init__(self, keys, orders, protected):
        if keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype != torch.long or keys.device.type != 'cpu':
            raise ValueError('Keys must be CPU long [rows,4]')
        if orders.shape != (len(keys),) or orders.dtype != torch.long or orders.device.type != 'cpu':
            raise ValueError('Orders must be CPU long [rows]')
        self.maps = {n: {} for n in ORDERS}
        self.protected = set(protected)
        for row, (values, order) in enumerate(zip(keys.tolist(), orders.tolist()), start=1):
            if order not in ORDERS or any(i != -1 for i in values[order:]):
                raise ValueError('Invalid n-gram order or padding')
            ids = values[:order]
            if any(not 0 <= i <= 65535 or i in self.protected for i in ids):
                raise ValueError('Memory key contains a control or invalid token')
            packed = int(packed_windows(np.array(ids), order)[0])
            if packed in self.maps[order]:
                raise ValueError('Duplicate exact memory key')
            self.maps[order][packed] = row
        self.last_orders = None

    def encode_with_mask(self, input_ids):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.device.type != 'cpu':
            raise ValueError('Exact lookup requires CPU long [batch,sequence]')
        result = np.zeros(tuple(input_ids.shape), dtype=np.int64)
        matched = np.zeros_like(result)
        for b, ids in enumerate(input_ids.numpy()):
            # Visit longest first; shorter hits fill only still-unmatched positions.
            for order in reversed(ORDERS):
                packed = packed_windows(ids, order)
                hits = np.fromiter((self.maps[order].get(int(k), 0) for k in packed), dtype=np.int64, count=len(packed))
                positions = result[b, order-1:]
                choose = (positions == 0) & (hits != 0)
                positions[choose] = hits[choose]
                matched[b, order-1:][choose] = order
        self.last_orders = torch.from_numpy(matched)
        indices = torch.from_numpy(result)
        return indices, indices.ne(0)


class GraftMemory(nn.Module):
    """Frozen teacher bank, with an immutable zero row reserved for misses."""
    def __init__(self, values, keys, orders, protected):
        super().__init__()
        if (values.ndim != 2 or values.device.type != 'cpu' or not values.is_floating_point()
                or values.shape[0] != len(keys) or not len(keys) or not torch.isfinite(values).all()):
            raise ValueError('Invalid frozen teacher vectors')
        self.addressing = ExactSuffixAddressing(keys, orders, protected)
        weight = torch.cat((torch.zeros((1, values.shape[1]), dtype=values.dtype), values))
        self.tables = nn.ModuleList([nn.Embedding.from_pretrained(weight, freeze=True)])
        self.train(False)

    @property
    def d_mem(self):
        return self.tables[0].embedding_dim

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def forward(self, indices):
        if indices.ndim != 2 or indices.dtype != torch.long:
            raise ValueError('Expected long exact indices [batch,sequence]')
        return self.tables[0](indices)

    def shuffle_rows(self, seed):
        generator = torch.Generator().manual_seed(seed)
        table = self.tables[0]
        order = torch.randperm(table.num_embeddings-1, generator=generator) + 1
        with torch.no_grad():
            table.weight[1:].copy_(table.weight.detach()[order].clone())
        # Leave row zero and all lookup keys/hit masks intact.

    @classmethod
    def from_directory(cls, directory, tokenizer_path):
        config = json.loads((directory/'graft-config.json').read_text())
        if config.get('kind') != 'memory-grafting-additive-v1' or config['student_tokenizer_sha256'] != sha256(tokenizer_path):
            raise ValueError('Graft bank kind or recipient tokenizer mismatch')
        payload = torch.load(directory/'graft-bank.pt', weights_only=True, map_location='cpu')
        if set(payload) != {'keys','orders','values','frequencies'}:
            raise ValueError('Unexpected bank payload')
        if payload['values'].shape != (config['rows'], config['d_mem']) or payload['values'].dtype != torch.bfloat16:
            raise ValueError('Graft vector shape/dtype does not match its publication')
        if config['control_ids'] != sorted(control_ids(tokenizer_path)):
            raise ValueError('Graft controls differ from the recipient tokenizer')
        if payload['frequencies'].shape != (config['rows'],) or torch.any(payload['frequencies'] < 2):
            raise ValueError('Invalid training frequency evidence')
        return cls(payload['values'], payload['keys'], payload['orders'], config['control_ids'])


class GraftReader(TargetReader):
    """Gated teacher values plus a four-tap depthwise causal convolution.

    Zero initialization preserves initial calibration while allowing immediate
    convolution gradients. A separate control mask prevents cross-document
    convolution; the exact-hit mask prevents corrections at memory misses.
    Online cached decoding is deliberately outside this pilot's scope.
    """
    def __init__(self, hidden_size=1024, memory_dim=2048, **kwargs):
        super().__init__(hidden_size, memory_dim, **kwargs)
        self.short_conv = nn.Parameter(torch.zeros(4, hidden_size))
        self.boundary_mask = None

    def _unscaled(self, hidden, memory, active):
        contribution, gates = super()._unscaled(hidden, memory, active)
        if (active is None or self.boundary_mask is None or self.boundary_mask.shape != active.shape
                or self.boundary_mask.dtype != torch.bool or self.boundary_mask.device != active.device):
            raise ValueError('Short convolution needs this batch\'s explicit control mask')
        document = self.boundary_mask.long().cumsum(1)
        combined = contribution * (1 + self.short_conv[0].to(contribution.dtype))
        length = hidden.shape[1]
        for lag in range(1, min(4, length)):
            allowed = active[:, lag:] & document[:, lag:].eq(document[:, :-lag])
            shifted = contribution[:, :-lag] * self.short_conv[lag].to(contribution.dtype)
            shifted = shifted * allowed.unsqueeze(-1)
            combined = combined + torch.nn.functional.pad(shifted, (0,0,lag,0))
        return combined, gates

    @torch.no_grad()
    def calibrate_residual_scale(self, hidden, memory, active=None, target_ratio=.005):
        """Match whole-block residual RMS, including exact-lookup misses."""
        if active is None or not active.any() or not 0 < target_ratio <= .1:
            raise ValueError('Graft calibration needs active hits and a small positive target')
        contribution, _ = self._unscaled(hidden, memory, active)
        rms_h = hidden.float().square().mean().sqrt()
        rms_delta = contribution.float().square().mean().sqrt()
        if not torch.isfinite(rms_h+rms_delta) or min(rms_h, rms_delta) <= 0:
            raise ValueError('Graft calibration requires finite nonzero RMS')
        scale = float(target_ratio*rms_h/rms_delta)
        if not math.isfinite(scale) or not 0 < scale <= 100:
            raise ValueError('Graft calibration requires extreme amplification')
        self.residual_scale.fill_(scale)
        return scale


def encoding_batch(directory, name, rows, width, compute):
    """Durable tensor-only batch; commit its checksum receipt last.

    A killed writer can leave an uncommitted file, which is recomputed. A
    committed batch with changed bytes or row identities fails closed.
    The caller binds the directory to the verified global encoding plan.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path, receipt = directory/(name+'.pt'), directory/(name+'.json')
    reused = receipt.exists()
    if reused:
        record = json.loads(receipt.read_text())
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError('Saved encoding batch checksum mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if set(payload) != {'rows', 'values'} or not torch.equal(payload['rows'], rows):
            raise RuntimeError('Saved encoding batch row identities changed')
        values = payload['values']
    else:
        values = compute()
    if (values.device.type != 'cpu' or values.dtype != torch.bfloat16
            or values.shape != (len(rows), width) or not torch.isfinite(values).all()):
        raise RuntimeError('Invalid teacher encoding batch')
    if not reused:
        temporary = path.with_suffix('.pt.partial')
        with temporary.open('wb') as handle:
            torch.save({'rows':rows, 'values':values}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        atomic_json(receipt, {'bytes':path.stat().st_size, 'sha256':sha256(path)})
    return values, reused


def build_bank(checkpoint, token_dir, teacher_dir, destination, resume=False):
    """One admitted offline pass: train-only mining, teacher encoding, atomic bank.

    Fixed limits keep this construction small: 4.19M sampled training tokens,
    at most 65,536 rows, teacher layer 24, and one hour including downloads/encoding.
    The caller must hold the project GPU lock and verify that CUDA is idle.
    """
    from tokenizers import Tokenizer
    from .graft_teacher import load_teacher_prefix, SOURCE_LAYER
    started = time.monotonic()
    work = destination.with_name(destination.name+'.partial')
    if destination.exists() or (work.exists() and not resume):
        raise ValueError('Choose a fresh graft-bank destination')
    if resume and not (work/'encoding-plan.json').is_file():
        raise ValueError('No resumable encoding plan; archive the legacy partial and choose a fresh attempt')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 20*2**30 + 1536*2**20:
        raise RuntimeError('Bank build needs 1.5GiB plus the untouched 20GiB reserve')
    work.mkdir(parents=True, exist_ok=resume)
    def report(phase, **extra):
        if time.monotonic()-started > 3600:
            raise RuntimeError('Offline bank build exceeded its one-hour budget')
        record = {'phase':phase,'elapsed_seconds':time.monotonic()-started,**extra}
        atomic_json(work/'build-status.json',record)
        print(json.dumps(record),flush=True)
    student_path = checkpoint/'tokenizer.json'
    teacher_path = teacher_dir/'tokenizer.json'
    student, teacher = Tokenizer.from_file(str(student_path)), Tokenizer.from_file(str(teacher_path))
    protected, teacher_protected = control_ids(student_path), control_ids(teacher_path, extra=())
    teacher_token_config = json.loads((teacher_dir/'tokenizer_config.json').read_text())
    bos = teacher_token_config.get('bos_token')
    if isinstance(bos, dict):
        bos = bos.get('content')
    bos_id = teacher.token_to_id(bos) if bos else None
    teacher_prefix = () if bos_id is None else (bos_id,)
    parent = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
    parent.pop('optimizer')
    summary_sha = sha256(token_dir/'manifest.json')
    chunks = sorted(n for n in parent['data']['known'] if n not in parent['validation_shards'])
    rng = np.random.default_rng(42)
    chosen = rng.permutation(chunks)[:64].tolist()
    counters = {n:Counter() for n in ORDERS}
    samples = []
    for name in chosen:
        manifest_path = token_dir/name/'manifest.json'
        if sha256(manifest_path) != parent['data']['known'][name]:
            raise RuntimeError('Training chunk manifest changed')
        metadata = json.loads(manifest_path.read_text())
        source = token_dir/name/'train.bin'
        if (metadata['tokenizer_sha256'] != sha256(student_path)
                or source.stat().st_size != metadata['splits']['train']['tokens']*2
                or sha256(source) != metadata['splits']['train']['sha256']):
            raise RuntimeError('Bank mining requires verified training bytes and tokenizer')
        values = np.memmap(source,dtype='<u2',mode='r')
        length = min(65536,len(values))
        offset = int(rng.integers(max(1,len(values)-length+1)))
        count_ngrams(values[offset:offset+length], protected, counters)
        samples.append({'chunk':name,'offset':offset,'tokens':length,
            'train_sha256':metadata['splits']['train']['sha256']})
        report('mining', chunks=len(samples), scanned_tokens=sum(s['tokens'] for s in samples))
    selected = select_keys(counters,student,teacher,teacher_protected,teacher_prefix=teacher_prefix)
    del counters
    config = json.loads((teacher_dir/'config.json').read_text())
    receipt = json.loads((teacher_dir/'source.json').read_text())
    plan = {'selected':selected, 'samples':samples, 'token_summary_sha256':summary_sha,
        'student_tokenizer_sha256':sha256(student_path), 'teacher_tokenizer_sha256':sha256(teacher_path),
        'teacher_repo':receipt['repo'], 'teacher_revision':receipt['revision'],
        'teacher_source_layer':SOURCE_LAYER, 'builder_sha256':sha256(Path(__file__)),
        'runtime':{'torch':str(torch.__version__), 'transformers':version('transformers'),
                   'tokenizers':version('tokenizers'), 'safetensors':version('safetensors')},
        'loader_sha256':sha256(Path(__file__).with_name('graft_teacher.py'))}
    plan_hash = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    plan_path = work/'encoding-plan.json'
    if resume:
        if json.loads(plan_path.read_text())['sha256'] != plan_hash:
            raise RuntimeError('Saved encoding plan differs from verified inputs or implementation')
    else:
        atomic_json(plan_path, {'sha256':plan_hash,'rows':len(selected), 'parent_step':parent['step']})
    report('loading_teacher', rows=len(selected))
    torch.cuda.reset_peak_memory_stats()
    scratch = work/f'teacher-scratch-{time.time_ns()}'
    model, teacher_identity = load_teacher_prefix(teacher_dir,scratch,report)
    hidden_size = config['text_config']['hidden_size']
    groups = defaultdict(list)
    for row, (_, _, ids) in enumerate(selected):
        groups[len(ids)].append((row,ids))
    vectors = torch.empty((len(selected), hidden_size), dtype=torch.bfloat16)
    encoded = reused_rows = 0
    with torch.inference_mode():
        for length in sorted(groups):
            entries = groups[length]
            for start in range(0,len(entries),64):
                batch = entries[start:start+64]
                rows = torch.tensor([e[0] for e in batch])
                def compute():
                    ids = torch.tensor([e[1] for e in batch],device='cuda')
                    output = model(input_ids=ids,use_cache=False)
                    return output.last_hidden_state[:,-1].detach().cpu()
                hidden, reused = encoding_batch(work/'encoded-batches',
                    f'length-{length:02d}-offset-{start:06d}',rows,hidden_size,compute)
                vectors[rows] = hidden
                encoded += len(batch)
                reused_rows += len(batch) if reused else 0
                if encoded == len(selected) or encoded % 4096 < len(batch):
                    report('encoding', encoded_rows=encoded, rows=len(selected), reused_rows=reused_rows)
    teacher_identity['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    keys = torch.full((len(selected),4),-1,dtype=torch.long)
    for row,(ids,_,_) in enumerate(selected):
        keys[row,:len(ids)] = torch.tensor(ids)
    orders = torch.tensor([len(e[0]) for e in selected])
    frequencies = torch.tensor([e[1] for e in selected])
    # Geometry diagnostics are descriptive, not a quality-based selection rule.
    sample = vectors[torch.randperm(len(vectors),generator=torch.Generator().manual_seed(42))[:1024]].float()
    rms = vectors.float().square().mean(1).sqrt()
    cosine = torch.nn.functional.cosine_similarity(sample[:len(sample)//2],sample[len(sample)//2:2*(len(sample)//2)],dim=-1)
    centered = sample-sample.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered@centered.T/len(sample)).clamp_min(0)
    geometry = {'row_rms_mean':float(rms.mean()),'row_rms_min':float(rms.min()),
        'row_rms_max':float(rms.max()),'random_pair_cosine_mean':float(cosine.mean()),
        'centered_pc1_fraction':float(eigenvalues[-1]/eigenvalues.sum().clamp_min(1e-12)),
        'centered_participation_rank':float(eigenvalues.sum().square()/eigenvalues.square().sum().clamp_min(1e-12))}
    report('publishing', rows=len(selected), geometry=geometry)
    if sha256(token_dir/'manifest.json') != summary_sha:
        raise RuntimeError('Token inventory changed during bank construction')
    with (work/'graft-bank.pt').open('wb') as handle:
        torch.save({'keys':keys,'orders':orders,'values':vectors,'frequencies':frequencies},handle)
        handle.flush()
        os.fsync(handle.fileno())
    bank_config = {'kind':'memory-grafting-additive-v1','rows':len(selected),'d_mem':hidden_size,
        'dtype':'bfloat16','student_tokenizer_sha256':sha256(student_path),
        'teacher_tokenizer_sha256':sha256(teacher_path),'control_ids':sorted(protected),
        'teacher_source_layer':SOURCE_LAYER,'teacher_bos_prefix':list(teacher_prefix),'orders':list(ORDERS),
        'row_budgets':list(ROW_BUDGETS),'rows_per_order':{str(n):int((orders==n).sum()) for n in ORDERS},
        'scan_tokens':sum(s['tokens'] for s in samples),'samples':samples,'geometry':geometry,
        'token_summary_sha256':summary_sha,'parent_step':parent['step'],'parent_tokens':parent['tokens'],
        'builder_sha256':sha256(Path(__file__)),'teacher':teacher_identity,
        'encoding_plan_sha256':plan_hash, 'resumed_encoding_rows':reused_rows}
    atomic_json(work/'graft-config.json',bank_config)
    construction = work/'construction-source'
    construction.mkdir(exist_ok=resume)
    shutil.copy2(Path(__file__),construction/'grafting.py')
    shutil.copy2(Path(__file__).with_name('graft_teacher.py'),construction/'graft_teacher.py')
    files = {name:{'bytes':(work/name).stat().st_size,'sha256':sha256(work/name)}
        for name in ('graft-bank.pt','graft-config.json')}
    atomic_json(work/'source.json',{'repo':receipt['repo'],'revision':receipt['revision'],
        'kind':bank_config['kind'],'files':files,'teacher_config_sha256':sha256(teacher_dir/'config.json')})
    report('completed', rows=len(selected), bytes=files['graft-bank.pt']['bytes'])
    work.rename(destination)
    # The committed final bank replaces the resumable batch copies.
    shutil.rmtree(destination/'encoded-batches')
    return bank_config
