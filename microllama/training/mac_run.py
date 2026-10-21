"""Bounded, single-MPS continuation with pinned data, validation and restart.

prepare is offline and makes a proposed plan. qualify exercises the real 8K
shape on isolated state. train requires that hardware/source-bound qualification
and an explicit finite target; no diagnostic checkpoint is a production parent.
"""
from __future__ import annotations

import argparse
import copy
import gc
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..data_io.blocks import TokenBlocks
from ..data_io.growth_reader import load_catalog
from ..data_io.mac_staging import ArchiveStager, payload_files, validate_reader_state
from .engram import load_engram_model
from .mac import (FINAL_MANIFEST, FINAL_STEP, FINAL_TOKENS, cpu_tree,
                  install_activation_checkpointing, install_mps_engram_hash_adapter,
                  install_mps_padding_adapter, install_mps_embedding_reduction_adapter,
                  load_final, require_production_parent,
                  restore_adam, sha256, update, write_json)
from .objective import chunked_loss, target_mask
from .mac_attention import install_tiled_attention

SCHEMA = 'microllama-mac-run-v1'
TOKENIZER = 'df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd'
ARCHIVES_SHA = 'e4b45c090459239492807fb1394cadd220a4d7cdd2b13e2c3fb9e3f1f132b1f4'
FILES_SHA = '1b2c644a53c38fd43aa999ec26d02379bbbb1a96feeae1d9e7e1f18ab8072d0c'
FINAL_LOSSES = dict(original=2.791546036133291, edu=2.7926703418413155,
                    dclm=2.7903212720533013, cosmopedia=1.718090737383364,
                    finemath=1.8793326959999062)
SOURCE_FILES = ('training/mac.py', 'training/mac_run.py', 'training/engram.py',
                'training/objective.py', 'training/optimizers.py',
                'data_io/mac_staging.py', 'data_io/growth_reader.py', 'data_io/blocks.py',
                'training/mac_tracking.py',
                'training/mac_attention.py',
                '../scripts/mac/qualify_mps.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_bindings():
    package = Path(__file__).resolve().parents[1]
    return {name: sha256(package / name) for name in SOURCE_FILES}


def hardware():
    def sysctl(name):
        return subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    return {'chip': sysctl('machdep.cpu.brand_string'), 'memory_bytes': int(sysctl('hw.memsize')),
            'macos': platform.mac_ver()[0], 'machine': platform.machine(),
            'torch': str(torch.__version__),
            'transformers': __import__('transformers').__version__}


def external(path):
    root = Path('/Volumes/TrainData')
    path = Path(path).resolve()
    if not root.is_mount() or root not in path.parents:
        raise ValueError('TrainData must be mounted; refusing large internal-SSD output')
    return path


@contextmanager
def device_lock(root):
    path = external(root) / '.mps-training.lock'
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another Mac trainer/qualifier owns the device lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state(device):
    return {'cpu': torch.get_rng_state(),
            'mps': torch.mps.get_rng_state() if device.type == 'mps' else None}


def restore_rng(saved, device):
    torch.set_rng_state(saved['cpu'].cpu())
    if device.type == 'mps':
        if saved['mps'] is None:
            raise ValueError('Missing MPS restart RNG')
        torch.mps.set_rng_state(saved['mps'].cpu())


def tensor_hash(value):
    array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def state_hashes(model, optimizer):
    return {'model': {name: tensor_hash(value) for name, value in model.state_dict().items()},
            'adam': {str(index): {key: tensor_hash(value) for key, value in slot.items()}
                     for index, slot in optimizer.state_dict()['state'].items()}}


def validation_spec(parent, catalog, inventory, initial_root):
    result = {'sequence_length': 8192, 'seed': 42, 'blocks': 128, 'minimum_valid_targets': 1_000_000,
              'chunk_size': parent['config']['loss_chunk_size'], 'tokenizer_sha256': TOKENIZER,
              'historical_baseline': parent['growth_validation']['baseline'],
              'final_reference': FINAL_LOSSES, 'sources': {}}
    selected = {'original': {name: inventory['fineweb-edu-initial'][name]['manifest.json']['sha256']
                             for name in parent['validation_shards']},
                **parent['growth_validation']['sources']}
    if set(selected) != set(FINAL_LOSSES):
        raise ValueError('Frozen validation source set differs')
    for source, chunks in selected.items():
        archive_source = 'fineweb-edu-initial' if source == 'original' else source
        directory = initial_root if source == 'original' else catalog['sources'][source]['directory']
        for name, expected in chunks.items():
            if inventory[archive_source][name]['manifest.json']['sha256'] != expected:
                raise ValueError('Frozen validation manifest differs from public inventory')
        result['sources'][source] = {'archive_source': archive_source, 'directory': str(directory),
                                      'chunks': chunks}
    return result


def validation_chunks(spec):
    return [(entry['archive_source'], name) for entry in spec['sources'].values()
            for name in entry['chunks']]


def metadata_coverage(catalog, data, target):
    """Use the original coverage policy without opening missing token files."""
    from ..data_io.growth_reader import MixtureTokenBlocks
    validate_reader_state(catalog, data)
    view = SimpleNamespace(parent_tokens=data['parent_tokens'], catalog=catalog,
        targets=copy.deepcopy(data['source_targets']), policy_origin=copy.deepcopy(data.get('policy_origin')),
        weights={name: entry['weight'] for name, entry in catalog['recipe']['sources'].items()},
        sequence_length=catalog['recipe']['sequence_length'])
    return MixtureTokenBlocks.coverage(view, target)


class FrozenValidation:
    """Same ordered heldout blocks; validation never changes training RNG/cursors."""
    def __init__(self, spec, stager=None):
        self.spec = copy.deepcopy(spec)
        self.progress = lambda value: None
        if stager:
            stager.stage_chunks(validation_chunks(spec))
        self.readers, self.identities = {}, {}
        for source, entry in spec['sources'].items():
            root = Path(entry['directory'])
            identities = []
            for name, expected in entry['chunks'].items():
                folder = root / name
                manifest, binary = folder / 'manifest.json', folder / 'validation.bin'
                if folder.is_symlink() or manifest.is_symlink() or binary.is_symlink():
                    raise ValueError('Frozen validation outputs must be regular files/directories')
                meta = json.loads(manifest.read_text())
                info = meta['splits']['validation']
                if (sha256(manifest) != expected or meta['tokenizer_sha256'] != spec['tokenizer_sha256']
                        or binary.stat().st_size != info['tokens'] * 2
                        or sha256(binary) != info['sha256']):
                    raise ValueError('Frozen validation payload identity differs')
                identities.extend((path, path.stat().st_size, path.stat().st_mtime_ns)
                                  for path in (manifest, binary))
            reader = TokenBlocks(root, spec['sequence_length'], split='validation', seed=spec['seed'],
                                 shards=list(entry['chunks']))
            if reader.num_blocks < spec['blocks']:
                raise ValueError('Frozen validation would repeat blocks')
            self.readers[source], self.identities[source] = reader, identities

    def __call__(self, model):
        device = next(model.parameters()).device
        saved_rng, was_training = rng_state(device), model.training
        result = {}
        model.eval()
        try:
            with torch.no_grad():
                for source, reader in self.readers.items():
                    started = time.monotonic()
                    self.progress({'phase': 'validating', 'source': source, 'blocks_done': 0,
                                   'blocks_total': self.spec['blocks']})
                    for path, size, mtime in self.identities[source]:
                        if path.stat().st_size != size or path.stat().st_mtime_ns != mtime:
                            raise ValueError('Frozen validation file changed during the run')
                    reader.epoch, reader.cursor = 0, 0
                    reader._shuffle()
                    nll, count, rows_hash = 0., 0, hashlib.sha256()
                    for index in range(self.spec['blocks']):
                        values = reader.next_batch(1)
                        rows_hash.update(values.astype('<u2').tobytes())
                        batch = torch.as_tensor(values, device=device)
                        x, y = batch[:, :-1], batch[:, 1:]
                        mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
                        loss = chunked_loss(model, x, y, self.spec['chunk_size'], loss_mask=mask, reduction='sum')
                        nll += float(loss.cpu())
                        count += int(mask.sum().cpu())
                        if (index + 1) % 8 == 0 or index + 1 == self.spec['blocks']:
                            self.progress({'phase': 'validating', 'source': source, 'blocks_done': index + 1,
                                'blocks_total': self.spec['blocks'], 'seconds': time.monotonic() - started})
                    if count < self.spec['minimum_valid_targets'] or not math.isfinite(nll) or nll <= 0:
                        raise ValueError('Invalid full frozen validation outcome')
                    result[source] = {'loss': nll / count, 'valid_targets': count,
                        'blocks': self.spec['blocks'], 'ordered_rows_sha256': rows_hash.hexdigest(),
                        'inventory_fingerprint': reader.fingerprint}
        finally:
            model.train(was_training)
            restore_rng(saved_rng, device)
        return result


def check_losses(measured, reference, tolerance, *, absolute=False):
    if not math.isfinite(tolerance) or tolerance <= 0 or set(measured) != set(reference):
        raise ValueError('Invalid validation guard/reference')
    for source, expected in reference.items():
        actual = measured[source]['loss']
        difference = abs(actual - expected) if absolute else actual - expected
        if not math.isfinite(actual) or actual <= 0 or difference > tolerance:
            raise RuntimeError(f'{source} validation differs from reference: {actual} versus {expected}')


def stager_for(plan, max_download_bytes):
    return ArchiveStager(plan['catalog'], plan['archives'], plan['files'], plan['archive_cache'],
        max_download_bytes=max_download_bytes, existing_archive_roots=plan['existing_archive_roots'],
        extra_source_roots={'fineweb-edu-initial': plan['initial_validation_root']})


def prepare(args):
    from .distributed import verify_checkpoint
    checkpoint = external(args.checkpoint)
    output = external(args.output)
    if output.exists():
        raise ValueError('Use a new plan path; never overwrite a reviewed plan')
    if sha256(checkpoint / 'checkpoint_manifest.json') != FINAL_MANIFEST:
        raise ValueError('Only the immutable final50B parent is admitted')
    verify_checkpoint(checkpoint)
    if sha256(args.archives) != ARCHIVES_SHA or sha256(args.files) != FILES_SHA:
        raise ValueError('Public data indexes differ from the verified recovery pins')
    parent = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
    if parent['tokens'] != FINAL_TOKENS or parent['step'] != FINAL_STEP:
        raise ValueError('Final parent counters differ')
    catalog = load_catalog(args.catalog)
    validate_reader_state(catalog, parent['data'])
    if (type(args.target_tokens) is not int or args.target_tokens <= FINAL_TOKENS
            or args.target_tokens > catalog['target_tokens']):
        raise ValueError('An explicit finite target inside the derived catalog budget is required')
    coverage = metadata_coverage(catalog, parent['data'], args.target_tokens)
    if coverage['balanced_unconsumed_targets'] < args.target_tokens - FINAL_TOKENS + 131072:
        raise ValueError('Finite target requires a complete-update coverage margin')
    spec = validation_spec(parent, catalog, json.loads(Path(args.files).read_text()),
                           external(args.initial_validation_root))
    config = {**parent['config'], 'target_tokens': args.target_tokens,
              'run_name': 'mac-continuation-from-final50b', 'precision': 'float32',
              'gradient_checkpointing': bool(args.activation_checkpointing),
              'save_every': args.save_every, 'eval_every': args.eval_every}
    for name in ('save_every', 'eval_every'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('Positive checkpoint/validation intervals required')
    if not 0 < args.max_validation_increase <= .3:
        raise ValueError('Validation guard must be explicit, finite and at most .3 loss')
    plan = {'schema': SCHEMA, 'kind': 'proposed-plan', 'training_started': False,
        'target_authorization': 'requires explicit train invocation after qualification',
        'parent': str(checkpoint), 'parent_manifest_sha256': FINAL_MANIFEST,
        'parent_step': FINAL_STEP, 'parent_tokens': FINAL_TOKENS,
        'catalog': str(external(args.catalog)),
        'catalog_sha256': json.loads((Path(args.catalog) / 'manifest.json').read_text())['sha256'],
        'archives': str(external(args.archives)), 'files': str(external(args.files)),
        'archives_sha256': ARCHIVES_SHA, 'files_sha256': FILES_SHA,
        'archive_cache': str(external(args.archive_cache)),
        'initial_validation_root': str(external(args.initial_validation_root)),
        'existing_archive_roots': [str(Path(p).resolve()) for p in args.existing_archive_root],
        'config': config, 'validation': spec, 'max_validation_increase': args.max_validation_increase,
        'attention_query_tile': 256,
        'mps_memory_fraction': .8,
        'backend_reference_tolerance': .05, 'implementation': source_bindings(), 'coverage': coverage,
        'cuda_to_mps_changes': ['world4->1', 'BF16 compute->FP32', 'fused->unfused AdamW',
                              'new explicitly seeded MPS RNG', 'CPU int64 token hashing',
                              'corrected input PAD gradient', 'ordered CPU embedding gradient reduction',
                              'optional decoder recomputation']}
    stager = stager_for(plan, 0)
    current = stager.estimate_current(parent['data'])['current_chunks']
    chunks = [*current.items(), *validation_chunks(spec)]
    plan['initial_transfer_plan'] = stager.estimate_chunks(chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return {'plan': str(output), 'plan_sha256': sha256(output),
            'target_tokens': args.target_tokens, 'download_bytes_needed': plan['initial_transfer_plan']['download_bytes_needed'],
            'training_started': False}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan.get('schema') != SCHEMA or plan.get('kind') != 'proposed-plan'
            or plan['parent_manifest_sha256'] != FINAL_MANIFEST
            or plan['implementation'] != source_bindings()):
        raise ValueError('Plan schema, parent or implementation differs; prepare a reviewed new plan')
    if (plan['archives_sha256'] != ARCHIVES_SHA or plan['files_sha256'] != FILES_SHA
            or sha256(plan['archives']) != ARCHIVES_SHA or sha256(plan['files']) != FILES_SHA):
        raise ValueError('Data indexes changed')
    if json.loads((Path(plan['catalog']) / 'manifest.json').read_text())['sha256'] != plan['catalog_sha256']:
        raise ValueError('Derived catalog changed')
    config = plan['config']
    parent_path = external(plan['parent'])
    if sha256(parent_path / 'checkpoint_manifest.json') != FINAL_MANIFEST:
        raise ValueError('Pinned parent manifest changed')
    manifest = json.loads((parent_path / 'checkpoint_manifest.json').read_text())
    for name in ('run_config.json', 'training_state.pt'):
        if sha256(parent_path / name) != manifest['files'][name]['sha256']:
            raise ValueError('Pinned parent restart metadata changed')
    original = json.loads((parent_path / 'run_config.json').read_text())
    allowed = {'target_tokens', 'run_name', 'precision', 'gradient_checkpointing', 'save_every', 'eval_every'}
    if set(config) != set(original) or any(config[key] != original[key] for key in set(original) - allowed):
        raise ValueError('Mac plan changed a preserved training recipe field')
    if (type(config['target_tokens']) is not int or config['target_tokens'] <= FINAL_TOKENS
            or type(config['gradient_checkpointing']) is not bool
            or any(type(config[key]) is not int or config[key] < 1 for key in ('save_every', 'eval_every'))
            or not math.isfinite(plan['max_validation_increase']) or not 0 < plan['max_validation_increase'] <= .3
            or plan['backend_reference_tolerance'] != .05):
        raise ValueError('Invalid finite target, memory policy or validation guard')
    if plan.get('attention_query_tile') != 256 or plan.get('mps_memory_fraction') != .8:
        raise ValueError('Unqualified attention tile policy')
    if (config['sequence_length'] != 8192 or config['micro_batch_size'] != 1
            or config['gradient_accumulation_steps'] != 16 or config['precision'] != 'float32'
            or config['learning_rate'] != .0003 or config['loss_policy'] != 'document-boundaries-v2'
            or config['schedule'] != 'constant-after-warmup-v1'):
        raise ValueError('Unqualified production batch, objective or optimizer recipe')
    parent = torch.load(parent_path / 'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
    expected = validation_spec(parent, load_catalog(plan['catalog']), json.loads(Path(plan['files']).read_text()),
                               external(plan['initial_validation_root']))
    if plan['validation'] != expected:
        raise ValueError('Mac plan changed the frozen validation protocol or inventory')
    for entry in load_catalog(plan['catalog'])['sources'].values():
        external(entry['directory'])
    return plan


def materialize(plan, parent_state, budget):
    stager = stager_for(plan, budget)
    current = stager.estimate_current(parent_state['data'])['current_chunks']
    # Combined preflight prevents partial download before discovering the
    # validation archive budget is insufficient.
    stager.stage_chunks([*current.items(), *validation_chunks(plan['validation'])])
    reader = stager.reader(parent_state['data'])
    validation = FrozenValidation(plan['validation'], stager)
    before = copy.deepcopy(reader.state_dict())
    values = reader.next_batch(16)
    after = reader.state_dict()
    reader.load_state_dict(before)
    # A second independent original-reader instance must select identical rows
    # and counters after the topology handoff. No weights are updated here.
    from ..data_io.growth_reader import MixtureTokenBlocks
    reference = MixtureTokenBlocks(plan['catalog'], 8192, seed=42)
    reference.load_state_dict(before)
    expected = reference.next_batch(16)
    if not np.array_equal(values, expected) or after != reference.state_dict():
        raise ValueError('Global next-row/cursor parity failed')
    return reader, validation, stager


def publish(root, model, optimizer, state, *, plan_sha, kind='mac-production', reserve_bytes=5 * 2**30,
            assets_from=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"step-{state['step']:08d}"
    temporary = path.with_name(path.name + '.partial')
    if path.exists() or temporary.exists():
        raise ValueError('Never overwrite a checkpoint or a failed partial publication')
    allowance = sum(value.numel() * value.element_size() for value in model.parameters()) * 3 + 64 * 2**20
    if shutil.disk_usage(root).free < reserve_bytes + allowance:
        raise RuntimeError('Checkpoint publication needs allowance plus data-drive reserve')
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    asset_names = {'tokenizer.json', 'tokenizer_config.json', 'source_manifest.json', 'chat_template.jinja'}
    if assets_from is not None:
        for name in asset_names:
            shutil.copyfile(Path(assets_from) / name, temporary / name)
    write_json(temporary / 'run_config.json', state['config'])
    saved = {**state, 'schema': SCHEMA, 'kind': kind, 'plan_sha256': plan_sha,
             'parent_manifest_sha256': FINAL_MANIFEST, 'optimizer': optimizer.state_dict(),
             'rng': rng_state(next(model.parameters()).device)}
    saved = cpu_tree(saved)
    if any(not bool(torch.isfinite(value).all()) for slot in saved['optimizer']['state'].values() for value in slot.values()):
        raise RuntimeError('Nonfinite Adam state; refusing checkpoint publication')
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise RuntimeError('Nonfinite model state; refusing checkpoint publication')
    torch.save(saved, temporary / 'mac_state.pt')
    payloads = payload_files(temporary, {'config.json', 'generation_config.json', 'model.safetensors',
                                       'engram_config.json', 'mac_state.pt', 'run_config.json', *asset_names})
    for file in payloads.values():
        with file.open('rb') as stream:
            os.fsync(stream.fileno())
    files = {file.name: {'bytes': file.stat().st_size, 'sha256': sha256(file)}
             for file in payloads.values()}
    write_json(temporary / 'mac_manifest.json', {'schema': SCHEMA, 'kind': kind,
              'plan_sha256': plan_sha, 'step': state['step'], 'tokens': state['tokens'], 'files': files})
    temporary.rename(path)
    write_json(root / 'latest.json', {'checkpoint': path.name,
                                    'manifest_sha256': sha256(path / 'mac_manifest.json')})
    return path


def read_checkpoint(path, plan_sha, *, kind='mac-production'):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Restart checkpoint directory must not be a symlink')
    manifest = json.loads((path / 'mac_manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('kind') != kind or manifest.get('plan_sha256') != plan_sha:
        raise ValueError('Restart schema, kind or plan differs')
    required = {'model.safetensors', 'mac_state.pt', 'config.json', 'engram_config.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Incomplete restart manifest')
    if set(payload_files(path, {*manifest['files'], 'mac_manifest.json'})) != {*manifest['files'], 'mac_manifest.json'}:
        raise ValueError('Restart file set differs')
    for name, info in manifest['files'].items():
        file = path / name
        if (Path(name).name != name or file.is_symlink() or not file.is_file()
                or file.stat().st_size != info['bytes'] or sha256(file) != info['sha256']):
            raise ValueError(f'Restart payload hash differs: {name}')
    state = torch.load(path / 'mac_state.pt', map_location='cpu', weights_only=True)
    if (state['schema'] != SCHEMA or state['kind'] != kind or state['plan_sha256'] != plan_sha
            or state['parent_manifest_sha256'] != FINAL_MANIFEST
            or state['step'] != manifest['step'] or state['tokens'] != manifest['tokens']):
        raise ValueError('Restart state/manifest binding differs')
    if kind == 'mac-production':
        require_production_parent(state)
    return state


def restore_model(path, state, device, checkpointing):
    model = load_engram_model(path).float()
    model.set_attn_implementation('sdpa')
    install_mps_engram_hash_adapter(model)
    install_mps_padding_adapter(model)
    install_mps_embedding_reduction_adapter(model)
    install_tiled_attention(model)
    if checkpointing:
        install_activation_checkpointing(model)
    model.to(device)
    optimizer = restore_adam(model, state['optimizer'], state['config'])
    restore_rng(state['rng'], device)
    return model, optimizer


class TrainingLoop:
    """Atomic global row consumption, update-boundary stops and validated finals."""
    def __init__(self, model, optimizer, reader, validator, state, config, *, validation_reference,
                 max_validation_increase, save, report=lambda value: None, initial_checkpoint=None,
                 initial_validation=None):
        self.model, self.optimizer, self.reader, self.validator = model, optimizer, reader, validator
        self.state, self.config = copy.deepcopy(state), config
        self.validation_reference, self.max_validation_increase = validation_reference, max_validation_increase
        self.save, self.report, self.last_saved = save, report, initial_checkpoint
        self.last_saved_step = state['step'] if initial_checkpoint is not None else None
        self.initial_validation = initial_validation
        if state['tokens'] != state['data']['parent_tokens'] + sum(state['data']['source_targets'].values()):
            raise ValueError('Restart counters and saved reader state do not reconcile')
        if reader.state_dict() != state['data']:
            raise ValueError('Live reader differs from restart state')

    def evaluate(self):
        self.report({'phase': 'validation-started', 'step': self.state['step'], 'tokens': self.state['tokens']})
        if hasattr(self.validator, 'progress'):
            self.validator.progress = lambda value: self.report({**value, 'step': self.state['step'], 'tokens': self.state['tokens']})
        measured = self.validator(self.model)
        check_losses(measured, self.validation_reference, self.max_validation_increase)
        self.state['last_validation'] = {'step': self.state['step'], 'tokens': self.state['tokens'],
                                         'sources': measured}
        self.report({'phase': 'validation', **self.state['last_validation']})

    def checkpoint(self):
        self.state['data'] = copy.deepcopy(self.reader.state_dict())
        self.last_saved = str(self.save(self.model, self.optimizer, copy.deepcopy(self.state)))
        self.last_saved_step = self.state['step']

    def run(self, *, max_updates, max_seconds, stopped=lambda: False):
        if ((max_updates is not None and (type(max_updates) is not int or max_updates < 1))
                or (max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0))):
            raise ValueError('Execution bounds must be positive or explicitly unset')
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        if self.initial_validation is None:
            self.evaluate()  # Mandatory on every changed-weight restart.
        else:
            # First launch may reuse the exact same immutable parent/model
            # already measured by this plan's hardware-bound qualification.
            check_losses(self.initial_validation, self.validation_reference, self.max_validation_increase)
            self.state['last_validation'] = {'step': self.state['step'], 'tokens': self.state['tokens'],
                                             'sources': copy.deepcopy(self.initial_validation)}
            self.report({'phase': 'validation-reused-from-qualification', **self.state['last_validation']})
        updates = 0
        target = self.config['target_tokens']
        device = next(self.model.parameters()).device
        while (self.state['tokens'] < target and (max_updates is None or updates < max_updates)
               and (deadline is None or time.monotonic() < deadline) and not stopped()):
            before = copy.deepcopy(self.reader.state_dict())
            started = time.monotonic()
            self.report({'phase': 'training-update', 'step': self.state['step'],
                         'target_step': self.state['step'] + 1, 'tokens': self.state['tokens']})
            try:
                rows = self.reader.next_batch(self.config['gradient_accumulation_steps'])
                if rows.shape != (self.config['gradient_accumulation_steps'], self.config['sequence_length'] + 1):
                    raise ValueError('Global input batch shape differs')
                # Preserve CPU integer IDs; no vocabulary or objective changes.
                batches = ((torch.as_tensor(row[None, :-1], device=device),
                            torch.as_tensor(row[None, 1:], device=device)) for row in rows)
                measured = update(self.model, self.optimizer, batches,
                    grad_clip=self.config['grad_clip'], chunk_size=self.config['loss_chunk_size'],
                    progress=lambda value: self.report({'phase': 'training-microbatch', 'step': self.state['step'],
                        'target_step': self.state['step'] + 1, 'tokens': self.state['tokens'], **value}))
                after = self.reader.state_dict()
                tokens = after['parent_tokens'] + sum(after['source_targets'].values())
                if tokens - self.state['tokens'] != measured['valid_targets']:
                    raise RuntimeError('Reader and objective valid-target counts disagree')
            except BaseException:
                self.reader.load_state_dict(before)
                self.optimizer.zero_grad(set_to_none=True)
                # Never publish this live model after a failed/partial update.
                raise
            updates += 1
            self.state.update(step=self.state['step'] + 1, tokens=tokens,
                input_positions=self.state['input_positions'] + rows.shape[0] * self.config['sequence_length'],
                data=after, phase='training')
            self.report({'phase': 'training', 'step': self.state['step'], 'tokens': tokens,
                         **measured, 'seconds': time.monotonic() - started})
            final = tokens >= target
            if final or self.state['step'] % self.config['eval_every'] == 0:
                self.evaluate()
            if final or self.state['step'] % self.config['save_every'] == 0:
                self.state['phase'] = 'completed' if final else 'training'
                self.checkpoint()
        self.state['phase'] = 'completed' if self.state['tokens'] >= target else 'paused'
        if self.last_saved_step != self.state['step']:
            self.checkpoint()
        return {'phase': self.state['phase'], 'step': self.state['step'], 'tokens': self.state['tokens'],
                'updates': updates, 'checkpoint': self.last_saved}


def require_qualification(receipt, plan_sha, machine):
    if (receipt.get('schema') != SCHEMA or receipt.get('kind') != 'mac8k-qualification'
            or receipt.get('passed') is not True or receipt.get('plan_sha256') != plan_sha
            or receipt.get('hardware') != machine or receipt.get('implementation') != source_bindings()
            or receipt.get('sequence_length') != 8192 or receipt.get('microbatches') != 16
            or receipt.get('real_reader_restart_exact') is not True
            or receipt.get('full_frozen_validation_pass') is not True):
        raise ValueError('Production requires a matching real-data 8K qualification on this hardware')


def qualification(args, plan, plan_sha):
    """Two real updates versus save/reload, isolated from all production state."""
    device, output = torch.device('mps'), external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Qualify backward semantics on this machine first; a matching forward
    # loss alone would miss the PAD-input gradient issue found on the M1.
    import importlib.util
    path = Path(__file__).resolve().parents[2] / 'scripts/mac/qualify_mps.py'
    module_spec = importlib.util.spec_from_file_location('mac_backend_checks', path)
    backend = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(backend)
    backend_result = backend.run('all', 'cpu', 'corrected', output, attention_backend='tiled')
    write_json(output / 'backend-checks.json', backend_result)
    if not backend_result['passed']:
        raise RuntimeError('CPU/MPS forward/backward/restart qualification failed')
    model, optimizer, parent = load_final(plan['parent'], device)
    install_tiled_attention(model, plan['attention_query_tile'])
    del parent['optimizer']
    if plan['config']['gradient_checkpointing']:
        install_activation_checkpointing(model)
    reader, validator, stager = materialize(plan, parent, args.max_download_bytes)
    validator.progress = lambda value: write_json(output / 'progress.json', {'unix': time.time(), **value})
    baseline = validator(model)
    check_losses(baseline, plan['validation']['final_reference'], plan['backend_reference_tolerance'], absolute=True)
    state = {'step': parent['step'], 'tokens': parent['tokens'], 'input_positions': parent['input_positions'],
             'data': parent['data'], 'config': plan['config'], 'validation': plan['validation'],
             'phase': 'qualification-only', 'last_validation': {'step': parent['step'], 'sources': baseline}}
    rows_hashes, metrics = [], []
    saved = None
    for index in range(2):
        rows = reader.next_batch(16)
        rows_hashes.append(hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest())
        started = time.monotonic()
        measured = update(model, optimizer, ((torch.as_tensor(row[None, :-1], device=device),
                         torch.as_tensor(row[None, 1:], device=device)) for row in rows),
                         grad_clip=plan['config']['grad_clip'], chunk_size=plan['config']['loss_chunk_size'],
                         progress=lambda value: write_json(output / 'progress.json',
                             {'unix': time.time(), 'phase': 'qualification-update', 'update': index + 1, **value}))
        state.update(step=state['step'] + 1, tokens=state['tokens'] + measured['valid_targets'], data=reader.state_dict(),
                     input_positions=state['input_positions'] + 131072)
        measured.update(seconds=time.monotonic() - started,
                        allocated_bytes=torch.mps.current_allocated_memory(), driver_bytes=torch.mps.driver_allocated_memory())
        metrics.append(measured)
        print(json.dumps({'qualification_update': index + 1, **measured}), flush=True)
        if index == 0:
            saved = publish(output / 'restart-fixture', model, optimizer, state,
                            plan_sha=plan_sha, kind='qualification-only')
    expected, expected_cursor, expected_rng = state_hashes(model, optimizer), reader.state_dict(), rng_state(device)
    del model, optimizer
    gc.collect()
    torch.mps.empty_cache()
    restored = read_checkpoint(saved, plan_sha, kind='qualification-only')
    reader.load_state_dict(restored['data'])
    model, optimizer = restore_model(saved, restored, device, plan['config']['gradient_checkpointing'])
    del restored['optimizer']
    rows = reader.next_batch(16)
    if hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest() != rows_hashes[1]:
        raise RuntimeError('Next real rows differ after restart')
    update(model, optimizer, ((torch.as_tensor(row[None, :-1], device=device),
           torch.as_tensor(row[None, 1:], device=device)) for row in rows),
           grad_clip=plan['config']['grad_clip'], chunk_size=plan['config']['loss_chunk_size'])
    actual_rng = rng_state(device)
    actual = state_hashes(model, optimizer)
    restart_check = {'model_exact': actual['model'] == expected['model'],
        'adam_exact': actual['adam'] == expected['adam'],
        'reader_exact': reader.state_dict() == expected_cursor,
        'rng_exact': {key: torch.equal(expected_rng[key], actual_rng[key]) for key in ('cpu', 'mps')},
        'expected': expected, 'actual': actual}
    write_json(output / 'restart-comparison.json', restart_check)
    if (not restart_check['model_exact'] or not restart_check['adam_exact'] or not restart_check['reader_exact']
            or not all(restart_check['rng_exact'].values())):
        raise RuntimeError('Real-data model/Adam/cursor/RNG restart is not exact')
    measured = validator(model)
    check_losses(measured, {key: value['loss'] for key, value in baseline.items()}, plan['max_validation_increase'])
    receipt = {'schema': SCHEMA, 'kind': 'mac8k-qualification', 'passed': True,
        'hardware': hardware(), 'implementation': source_bindings(), 'plan_sha256': plan_sha,
        'sequence_length': 8192, 'microbatches': 16, 'real_reader_restart_exact': True,
        'full_frozen_validation_pass': True, 'baseline': baseline, 'after': measured,
        'metrics': metrics, 'ordered_batch_hashes': rows_hashes,
        'backend_checks_sha256': sha256(output / 'backend-checks.json'),
        'official_parent_tokens_advanced': 0, 'download_bytes_requested': stager.download_bytes}
    write_json(output / 'qualification.json', receipt)
    del model, optimizer, restored, reader, validator
    gc.collect()
    torch.mps.empty_cache()
    return receipt


def train(args, plan, plan_sha):
    receipt = json.loads(Path(args.qualification).read_text())
    require_qualification(receipt, plan_sha, hardware())
    device, output = torch.device('mps'), external(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'events.jsonl'
    if args.resume:
        state = read_checkpoint(external(args.resume), plan_sha)
        if state['config'] != plan['config'] or state['validation'] != plan['validation']:
            raise ValueError('Restart configuration or frozen validation differs')
        model, optimizer = restore_model(args.resume, state, device, plan['config']['gradient_checkpointing'])
        del state['optimizer']
    else:
        if (output / 'checkpoints' / 'latest.json').exists() or log.exists():
            raise ValueError('Existing run requires an explicit resume checkpoint')
        model, optimizer, parent = load_final(plan['parent'], device)
        install_tiled_attention(model, plan['attention_query_tile'])
        del parent['optimizer']
        if plan['config']['gradient_checkpointing']:
            install_activation_checkpointing(model)
        state = {'step': parent['step'], 'tokens': parent['tokens'], 'input_positions': parent['input_positions'],
                 'data': parent['data'], 'config': plan['config'], 'validation': plan['validation']}
    if state['tokens'] >= plan['config']['target_tokens']:
        raise ValueError('Finite target is already complete; no automatic continuation')
    reader, validator, stager = materialize(plan, state, args.max_download_bytes)
    reader.coverage(plan['config']['target_tokens'])
    from .mac_tracking import Tracker
    tracker = Tracker(output / 'tracking', plan_sha, {'plan_sha256': plan_sha,
        'parent_manifest_sha256': FINAL_MANIFEST, 'parent_tokens': FINAL_TOKENS,
        'config': plan['config'], 'hardware': hardware(), 'kind': 'new-mac-continuation'})
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    def report(value):
        value = {'unix': time.time(), **value}
        with log.open('a') as stream:
            stream.write(json.dumps(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        write_json(output / 'status.json', value)
        tracker.log(value)
        if getattr(args, 'worker_status_path', None):
            write_json(args.worker_status_path, {**value, 'pid': os.getpid(),
                'training_started': value.get('phase', '').startswith('training') or value.get('tokens', FINAL_TOKENS) > FINAL_TOKENS,
                'wandb_id': tracker.identity['id'], 'wandb_mode': tracker.mode})
        print(json.dumps(value), flush=True)
    baseline = {key: value['loss'] for key, value in receipt['baseline'].items()}
    loop = TrainingLoop(model, optimizer, reader, validator, state, plan['config'],
        validation_reference=baseline, max_validation_increase=plan['max_validation_increase'],
        save=lambda m, o, s: publish(output / 'checkpoints', m, o, s, plan_sha=plan_sha,
                                    assets_from=plan['parent']), report=report,
        initial_checkpoint=str(args.resume) if args.resume else None,
        initial_validation=None if args.resume else receipt['baseline'])
    failed = True
    try:
        result = loop.run(max_updates=args.max_updates, max_seconds=args.max_seconds, stopped=lambda: stopped)
        report({**result, 'download_bytes_requested': stager.download_bytes})
        failed = False
        return result
    except BaseException as error:
        report({'phase': 'failed', 'error': f'{type(error).__name__}: {error}',
                'last_published_checkpoint': loop.last_saved,
                'download_bytes_requested': stager.download_bytes})
        raise
    finally:
        tracker.finish(exit_code=1 if failed else 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def pipeline(args, plan, plan_sha):
    """Detached one-shot worker: qualify first, then a bounded continuation.

    No automatic retry, restart, dataset growth or target extension. A failure
    leaves its log, phase and all partial evidence for review.
    """
    root = external(args.output)
    root.mkdir(parents=True, exist_ok=False)
    deadline = None if args.max_seconds is None else time.monotonic() + args.max_seconds
    def status(value):
        write_json(root / 'worker-status.json', {'unix': time.time(), 'pid': os.getpid(), **value})
    status({'phase': 'qualifying', 'plan_sha256': plan_sha, 'training_started': False})
    from .mac_tracking import identity
    tracking = identity(root / 'training' / 'tracking', plan_sha, mode='auto')
    status({'phase': 'qualifying', 'plan_sha256': plan_sha, 'training_started': False,
            'new_wandb_id': tracking['id'], 'old_wandb_run_reused': False})
    try:
        args.output = root / 'qualification'
        receipt = qualification(args, plan, plan_sha)
        status({'phase': 'qualified', 'passed': receipt['passed'], 'training_started': False})
        args.qualification = root / 'qualification' / 'qualification.json'
        args.output, args.resume = root / 'training', None
        args.worker_status_path = root / 'worker-status.json'
        # This process shares one total transfer ceiling across the phases.
        args.max_download_bytes -= receipt['download_bytes_requested']
        status({'phase': 'starting-training', 'training_started': False})
        args.max_seconds = None if deadline is None else deadline - time.monotonic()
        if args.max_seconds is not None and args.max_seconds <= 0:
            result = {'phase': 'paused-before-training', 'updates': 0,
                      'reason': 'Wall-clock allowance consumed by qualification'}
            status(result)
            return result
        result = train(args, plan, plan_sha)
        status({**result, 'training_started': result['updates'] > 0})
        return result
    except BaseException as error:
        status({'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare', help='Offline proposal and aggregate transfer estimate')
    for name in ('checkpoint', 'catalog', 'archives', 'files', 'initial-validation-root', 'archive-cache', 'output'):
        prepare_parser.add_argument('--' + name, type=Path, required=True)
    prepare_parser.add_argument('--target-tokens', type=int, required=True)
    prepare_parser.add_argument('--activation-checkpointing', action='store_true')
    prepare_parser.add_argument('--save-every', type=int, default=50)
    prepare_parser.add_argument('--eval-every', type=int, default=250)
    prepare_parser.add_argument('--max-validation-increase', type=float, default=.1)
    prepare_parser.add_argument('--existing-archive-root', type=Path, action='append', default=[])
    for name in ('qualify', 'train', 'pipeline'):
        sub = commands.add_parser(name)
        sub.add_argument('--plan', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--max-download-bytes', type=int, default=0)
        if name == 'train':
            sub.add_argument('--qualification', type=Path, required=True)
            sub.add_argument('--resume', type=Path)
        if name in ('train', 'pipeline'):
            sub.add_argument('--max-updates', type=int)
            sub.add_argument('--max-seconds', type=float)
            sub.add_argument('--until-stopped', action='store_true', help='Explicitly remove time/update pilot limits; retain data/quality/stop gates')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args), indent=2)); return
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('An available MPS GPU and disabled blanket fallback are required')
    if not math.isfinite(float(args.max_download_bytes)) or args.max_download_bytes < 0:
        raise ValueError('Nonnegative bounded transfer allowance required')
    if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') in ('0', '0.0'):
        raise ValueError('Mac training requires the MPS allocator safety limit')
    if args.command in ('train', 'pipeline'):
        if args.until_stopped:
            if args.max_updates is not None or args.max_seconds is not None:
                raise ValueError('Choose until-stopped or explicit execution bounds, not both')
        elif (args.max_updates is None or args.max_seconds is None or args.max_updates < 1
              or not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
            raise ValueError('Training requires explicit positive bounds or --until-stopped')
    plan = load_plan(args.plan)
    torch.mps.set_per_process_memory_fraction(plan['mps_memory_fraction'])
    root = Path(plan['archive_cache']).parent
    with device_lock(root):
        if args.command == 'pipeline':
            result = pipeline(args, plan, sha256(args.plan))
        elif args.command == 'qualify':
            result = qualification(args, plan, sha256(args.plan))
        else:
            result = train(args, plan, sha256(args.plan))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
