"""Qualified BF16/FP32-state MLX continuation, with exact native restarts.

Large artifacts and all training data stay on TrainData. The original Mac
checkpoint, source-frozen attempts and CUDA history are immutable parents.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
import torch

from . import mac_run as base
from .mac import FINAL_MANIFEST, sha256, write_json
from .mac_mlx import adam_from_torch, adam_update, loss, memory_indices, scaled_value_and_grad
from ..data_io.growth_reader import load_catalog
from ..data_io.mac_staging import payload_files, validate_reader_state

SCHEMA = 'microllama-mac-mlx-v1'
SOURCE_FILES = (*base.SOURCE_FILES, 'training/mac_mlx.py', 'training/mac_mlx_run.py',
                'training/mac_gradient_check.py', '../scripts/mac/bench_speed.py',
                '../tests/test_mac_mlx.py', '../tests/test_mac_mlx_run.py',
                '../tests/test_mac_speed.py', '../tests/test_mac_training.py')
ASSETS = ('config.json', 'generation_config.json', 'engram_config.json', 'tokenizer.json',
          'tokenizer_config.json', 'source_manifest.json', 'chat_template.jinja')


def bindings():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256(root / name) for name in SOURCE_FILES}


def hardware():
    return {**base.hardware(), 'mlx': importlib.metadata.version('mlx'),
            'mlx_metal': importlib.metadata.version('mlx-metal'), 'metal': mx.device_info(mx.gpu)}


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def restore_rng_key(value):
    # MLX0.32 uses a thread-local RandomState sentinel, not an assignable
    # Python list. A PRNG key is the two uint32 halves of a uint64 seed.
    key = np.asarray(value)
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ValueError('Unexpected MLX PRNG key layout')
    mx.random.seed((int(key[0]) << 32) | int(key[1]))
    if not np.array_equal(np.asarray(mx.random.state[0]), key):
        raise RuntimeError('MLX PRNG key restoration failed')


def signatures(engine, state):
    mx.eval(engine.params, engine.slots, mx.random.state)
    return {'model': {n: array_hash(engine.params[n]) for n in sorted(engine.params)},
            'adam': {n: {'m': array_hash(engine.slots[n]['m']), 'v': array_hash(engine.slots[n]['v']),
                          'step': engine.slots[n]['step']} for n in sorted(engine.slots)},
            'reader': base.digest(state['data']), 'rng': [array_hash(x) for x in mx.random.state]}


class Engine:
    def __init__(self, params, slots, policy, config, *, query_tile=512, chunk=256, extras=None):
        self.params = {n: params[n] for n in sorted(params)}
        self.slots, self.policy = slots, policy
        self.config, self.extras = config, extras or {}
        self.query_tile, self.chunk = query_tile, chunk
        self.table_sizes = [self.params[f'model.engram.{order}_tables.{h}.weight'].shape[0]
            for order in ['bigram', 'trigram'] for h in range(config['engram_hash_heads'])]
        self.gradient = scaled_value_and_grad(lambda p, rows, indices: loss(p, config, rows, indices,
            compute='bfloat16', query_tile=query_tile, chunk=chunk, checkpoint=True, reduction='sum'),
            compute='bfloat16', compiled=True, loss_scale=1)
        self.forward = mx.compile(lambda p, rows, indices: loss(p, config, rows, indices,
            compute='bfloat16', query_tile=query_tile, chunk=chunk, checkpoint=False, reduction='sum'))

    def indices(self, rows):
        return memory_indices(self.config, rows[:, :-1], self.table_sizes)

    def update(self, rows, report=lambda value: None):
        if rows.shape != (16, 8193):
            raise ValueError('Production requires original global16/full8K input')
        total, summed, accumulated = 0, 0., None
        started = time.monotonic()
        mx.reset_peak_memory()
        for index, row in enumerate(rows):
            batch = row[None]
            count = int(((batch[:, :-1] != 7) & (batch[:, :-1] != 0) & (batch[:, 1:] != 0)).sum())
            if count < 1:
                raise ValueError('No valid targets in microbatch')
            value, gradients = self.gradient(self.params, mx.array(batch), self.indices(batch))
            accumulated = {n: gradients[n] + (0 if accumulated is None else accumulated[n])
                           for n in sorted(gradients)}
            mx.eval(value, accumulated)
            total += count; summed += float(value)
            if not math.isfinite(summed):
                raise RuntimeError('Nonfinite summed training NLL')
            report({'microbatches_done': index + 1, 'microbatches_total': 16,
                    'valid_targets_so_far': total, 'seconds': time.monotonic() - started})
            del value, gradients
        gradients = {n: accumulated[n] / total for n in sorted(accumulated)}
        updated, moments, norm = adam_update(self.params, gradients, self.slots, self.policy)
        mx.eval(updated, moments, norm)
        if not math.isfinite(float(norm)):
            raise RuntimeError('Nonfinite global gradient norm')
        self.params, self.slots = updated, moments
        return {'loss': summed / total, 'valid_targets': total, 'grad_norm': float(norm),
                'seconds': time.monotonic() - started, 'peak_bytes': mx.get_peak_memory(),
                'valid_targets_per_second': total / (time.monotonic() - started)}


def import_mac_checkpoint(plan):
    path = Path(plan['parent'])
    if sha256(path / 'mac_manifest.json') != plan['parent_manifest_sha256']:
        raise ValueError('Pinned Mac parent manifest differs')
    saved = base.read_checkpoint(path, plan['parent_plan_sha256'])
    from .engram import EngramQwen3Config, EngramQwen3ForCausalLM
    from .optimizers import build_optimizer
    with torch.device('meta'):
        model = EngramQwen3ForCausalLM(EngramQwen3Config.from_pretrained(path, local_files_only=True))
    optimizer = build_optimizer(model, fused=False, memory_lr_multiplier=5)
    slots, policy = adam_from_torch(model, optimizer, saved['optimizer'])
    names = sorted(n for n, _ in model.named_parameters())
    loaded = mx.load(str(path / 'model.safetensors'), stream=mx.cpu)
    params = {n: loaded[n].astype(mx.float32) for n in names}
    extras = {n: v for n, v in loaded.items() if n not in params and n != 'lm_head.weight'}
    if set(extras) - {'model.engram.canonical_ids'}:
        raise ValueError('Unexpected nonparameter model tensors')
    config = json.loads((path / 'config.json').read_text())
    state = {k: copy.deepcopy(saved[k]) for k in ('step', 'tokens', 'input_positions', 'data')}
    state.update(schema=SCHEMA, kind='mac-production', config=plan['config'],
        validation=plan['validation'], final_cuda_manifest_sha256=FINAL_MANIFEST,
        imported_mac_manifest_sha256=plan['parent_manifest_sha256'], phase='ready')
    del loaded, saved, optimizer, model
    gc.collect()
    mx.random.seed(42)
    engine = Engine(params, slots, policy, config, query_tile=plan['query_tile'], chunk=plan['loss_chunk'], extras=extras)
    mx.eval(engine.params, engine.slots)
    return engine, state


def publish(root, engine, state, plan_sha, assets_from, *, kind='mac-production'):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{state['step']:08d}"
    partial = target.with_name(target.name + '.partial')
    if target.exists() or partial.exists():
        raise ValueError('Never overwrite a native checkpoint/failed partial')
    needed = sum(p.nbytes for p in engine.params.values()) * 3 + 128 * 2**20
    if shutil.disk_usage(root).free < needed + 20 * 2**30:
        raise RuntimeError('Native publication needs checkpoint allowance and20GiB reserve')
    for collection in (engine.params, {n: s['m'] for n, s in engine.slots.items()},
                       {n: s['v'] for n, s in engine.slots.items()}):
        checks = [mx.all(mx.isfinite(collection[n])) for n in sorted(collection)]
        mx.eval(checks)
        if not all(bool(x) for x in checks):
            raise RuntimeError('Nonfinite parameter or Adam moment; refusing publication')
    partial.mkdir()
    mx.save_safetensors(str(partial / 'model.safetensors'), {**engine.params, **engine.extras})
    mx.save_safetensors(str(partial / 'optimizer.safetensors'),
        {f'{slot}:{n}': engine.slots[n][slot] for n in sorted(engine.slots) for slot in ('m', 'v')})
    mx.save_safetensors(str(partial / 'rng.safetensors'), {str(i): x for i, x in enumerate(mx.random.state)})
    metadata = {**copy.deepcopy(state), 'schema': SCHEMA, 'kind': kind, 'plan_sha256': plan_sha,
                'policy': engine.policy, 'adam_steps': {n: s['step'] for n, s in engine.slots.items()},
                'parameter_shapes': {n: list(p.shape) for n, p in engine.params.items()},
                'extra_names': sorted(engine.extras), 'final_cuda_manifest_sha256': FINAL_MANIFEST}
    write_json(partial / 'state.json', metadata)
    for name in ASSETS:
        shutil.copyfile(Path(assets_from) / name, partial / name)
    write_json(partial / 'run_config.json', state['config'])
    expected = {*ASSETS, 'model.safetensors', 'optimizer.safetensors', 'rng.safetensors', 'state.json', 'run_config.json'}
    files = payload_files(partial, expected)
    for file in files.values():
        with file.open('rb') as stream: os.fsync(stream.fileno())
    manifest = {'schema': SCHEMA, 'kind': kind, 'plan_sha256': plan_sha, 'step': state['step'],
                'tokens': state['tokens'], 'files': {n: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for n, p in files.items()}}
    write_json(partial / 'mlx_manifest.json', manifest)
    partial.rename(target)
    write_json(root / 'latest.json', {'checkpoint': target.name, 'manifest_sha256': sha256(target / 'mlx_manifest.json')})
    return target


def restore(path, plan_sha, *, kind='mac-production'):
    path = Path(path)
    if path.is_symlink(): raise ValueError('Native checkpoint cannot be a symlink')
    manifest = json.loads((path / 'mlx_manifest.json').read_text())
    if manifest['schema'] != SCHEMA or manifest['kind'] != kind or manifest['plan_sha256'] != plan_sha:
        raise ValueError('Native checkpoint schema/kind/plan differs')
    names = set(manifest['files'])
    if not {*ASSETS, 'model.safetensors', 'optimizer.safetensors', 'rng.safetensors', 'state.json', 'run_config.json'} <= names:
        raise ValueError('Incomplete native checkpoint')
    if set(payload_files(path, names | {'mlx_manifest.json'})) != names | {'mlx_manifest.json'}:
        raise ValueError('Native checkpoint file set differs')
    for n, info in manifest['files'].items():
        p = path / n
        if Path(n).name != n or p.is_symlink() or p.stat().st_size != info['bytes'] or sha256(p) != info['sha256']:
            raise ValueError(f'Native checkpoint payload differs: {n}')
    state = json.loads((path / 'state.json').read_text())
    if (state['schema'] != SCHEMA or state['kind'] != kind or state['plan_sha256'] != plan_sha
            or state['step'] != manifest['step'] or state['tokens'] != manifest['tokens']
            or state['final_cuda_manifest_sha256'] != FINAL_MANIFEST):
        raise ValueError('Native state/manifest/provenance differs')
    model = mx.load(str(path / 'model.safetensors'), stream=mx.cpu)
    optimizer = mx.load(str(path / 'optimizer.safetensors'), stream=mx.cpu)
    param_names = set(state['parameter_shapes'])
    if set(model) != param_names | set(state['extra_names']) or set(optimizer) != {f'{s}:{n}' for n in param_names for s in ('m', 'v')}:
        raise ValueError('Native parameter/Adam identity differs')
    params = {n: model[n] for n in sorted(param_names)}
    slots = {n: {'m': optimizer[f'm:{n}'], 'v': optimizer[f'v:{n}'], 'step': state['adam_steps'][n]} for n in sorted(param_names)}
    for n, p in params.items():
        if list(p.shape) != state['parameter_shapes'][n] or p.dtype != mx.float32:
            raise ValueError('Master parameter shape/dtype differs')
        if any(slots[n][s].shape != p.shape or slots[n][s].dtype != mx.float32 for s in ('m', 'v')):
            raise ValueError('Adam moment shape/dtype differs')
    rng = mx.load(str(path / 'rng.safetensors'), stream=mx.cpu)
    if set(rng) != {'0'}:
        raise ValueError('Unexpected native RNG file keys')
    restore_rng_key(rng['0'])
    config = json.loads((path / 'config.json').read_text())
    engine = Engine(params, slots, state['policy'], config,
        query_tile=state['config']['attention_query_tile'], chunk=state['config']['loss_chunk_size'],
        extras={n: model[n] for n in state['extra_names']})
    mx.eval(engine.params, engine.slots, mx.random.state)
    return engine, state


class Validation:
    def __init__(self, spec, stager):
        self.original = base.FrozenValidation(spec, stager)
        self.spec, self.progress = spec, lambda value: None

    def __call__(self, engine):
        before_rng = [array_hash(x) for x in mx.random.state]
        result = {}
        for source, reader in self.original.readers.items():
            for path, size, mtime in self.original.identities[source]:
                if path.stat().st_size != size or path.stat().st_mtime_ns != mtime:
                    raise ValueError('Frozen validation payload changed')
            reader.epoch, reader.cursor = 0, 0; reader._shuffle()
            nll, count, rows_hash = 0., 0, hashlib.sha256(); start = time.monotonic()
            for index in range(self.spec['blocks']):
                rows = reader.next_batch(1); rows_hash.update(rows.astype('<u2').tobytes())
                value = engine.forward(engine.params, mx.array(rows), engine.indices(rows)); mx.eval(value)
                nll += float(value)
                count += int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
                if (index + 1) % 8 == 0 or index + 1 == self.spec['blocks']:
                    self.progress({'phase': 'validating', 'source': source, 'blocks_done': index + 1,
                                   'blocks_total': self.spec['blocks'], 'seconds': time.monotonic() - start})
            if count < self.spec['minimum_valid_targets'] or not math.isfinite(nll) or nll <= 0:
                raise RuntimeError('Incomplete/nonfinite frozen validation')
            result[source] = {'loss': nll / count, 'valid_targets': count, 'blocks': self.spec['blocks'],
                'ordered_rows_sha256': rows_hash.hexdigest(), 'inventory_fingerprint': reader.fingerprint}
        if before_rng != [array_hash(x) for x in mx.random.state]:
            raise RuntimeError('Validation consumed training RNG')
        return result


def advance(engine, reader, state, report=lambda value: None):
    before = copy.deepcopy(reader.state_dict())
    try:
        rows = reader.next_batch(16)
        metrics = engine.update(rows, report)
        after = reader.state_dict()
        tokens = after['parent_tokens'] + sum(after['source_targets'].values())
        if tokens - state['tokens'] != metrics['valid_targets']:
            raise RuntimeError('Reader and objective target counts differ')
    except BaseException:
        reader.load_state_dict(before)
        raise
    state.update(step=state['step'] + 1, tokens=tokens, data=after,
                 input_positions=state['input_positions'] + 131072, phase='training')
    metrics['rows_sha256'] = hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()
    return metrics


def make_plan(args):
    old = base.load_plan(args.base_plan)
    path, output = base.external(args.parent), base.external(args.output)
    if output.exists(): raise ValueError('Use a fresh reviewed MLX plan')
    state = base.read_checkpoint(path, sha256(args.base_plan))
    if state['config'] != old['config'] or state['validation'] != old['validation']:
        raise ValueError('Continuation recipe or frozen validation differs')
    if validate_reader_state(load_catalog(old['catalog']), state['data']) != state['tokens']:
        raise ValueError('Parent source accounting differs')
    config = {**state['config'], 'precision': 'bf16', 'attention_query_tile': 512,
              'run_name': 'm1-mlx-bf16-continuation'}
    plan = {'schema': SCHEMA, 'base_plan': str(args.base_plan.resolve()), 'base_plan_sha256': sha256(args.base_plan),
            'parent': str(path), 'parent_manifest_sha256': sha256(path / 'mac_manifest.json'),
            'parent_plan_sha256': sha256(args.base_plan), 'parent_step': state['step'], 'parent_tokens': state['tokens'],
            'config': config, 'validation': old['validation'], 'max_validation_increase': old['max_validation_increase'],
            'backend_reference_tolerance': old['backend_reference_tolerance'], 'query_tile': 512,
            'loss_chunk': old['config']['loss_chunk_size'],
            'implementation': bindings(), 'hardware': hardware(), 'memory_limit_bytes': 18 * 2**30,
            'retain_production_checkpoints': 2, 'max_download_bytes': 0,
            'execution_policy': 'until-user-stop-or-existing-data-quality-gate',
            'cpu_oracle_control': str(base.external(args.cpu_oracle_control)),
            'cpu_oracle_control_sha256': sha256(args.cpu_oracle_control),
            'precision_policy': 'BF16-compute-FP32-masters-moments-global-gradient-accumulation',
            'backward_policy': 'summed-NLL-then-single-global-FP32-normalization'}
    write_json(output, plan)
    return {'plan': str(output), 'sha256': sha256(output), 'training_started': False}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan['schema'] != SCHEMA or plan['implementation'] != bindings() or plan['hardware'] != hardware()
            or sha256(plan['base_plan']) != plan['base_plan_sha256']
            or plan['parent_plan_sha256'] != plan['base_plan_sha256']):
        raise ValueError('MLX plan source/hardware/base binding differs')
    old = base.load_plan(plan['base_plan'])
    allowed = {'precision', 'loss_chunk_size', 'attention_query_tile', 'run_name'}
    if (set(plan['config']) != set(old['config']) | {'attention_query_tile'}
            or any(plan['config'][n] != old['config'][n] for n in old['config'] if n not in allowed)
            or plan['config']['precision'] != 'bf16' or plan['query_tile'] != 512 or plan['loss_chunk'] != old['config']['loss_chunk_size']
            or plan['config']['loss_chunk_size'] != old['config']['loss_chunk_size'] or plan['config']['attention_query_tile'] != 512
            or plan['validation'] != old['validation'] or plan['backend_reference_tolerance'] != .05
            or plan['max_validation_increase'] != old['max_validation_increase'] or plan['max_download_bytes'] != 0
            or plan['retain_production_checkpoints'] != 2):
        raise ValueError('MLX plan changed a preserved data/quality/training policy')
    if sha256(plan['cpu_oracle_control']) != plan['cpu_oracle_control_sha256']:
        raise ValueError('Numerical oracle control binding differs')
    return plan, old


def check_cpu_oracle(engine, reader, plan, output, report):
    from .mac_gradient_check import full_gradient_check
    control = json.loads(Path(plan['cpu_oracle_control']).read_text())
    results_path = Path(control['results'])
    deadline = time.monotonic() + 1200
    while not results_path.exists():
        report({'phase': 'waiting-for-cpu-oracle', 'training_started': False})
        if time.monotonic() >= deadline:
            raise RuntimeError('CPU oracle not complete; preserve attempts and investigate')
        time.sleep(30)
    result = json.loads(results_path.read_text())
    if (sha256(results_path) != control['results_sha256']
            or result['parent_manifest_sha256'] != plan['parent_manifest_sha256']
            or result['parent_plan_sha256'] != plan['parent_plan_sha256']
            or result['length'] != 8192 or result['query_tile'] != plan['query_tile']
            or result['loss_chunk'] != plan['loss_chunk'] or result['precision'] != 'float32'
            or result['code_sha256'] != control['code_sha256'] or set(result['backends']) != {'cpu'}):
        raise ValueError('CPU oracle identity/source/policy differs')
    # An independent FP32 CPU calculation is an accuracy reference, not a
    # production precision setting. CPU BF16 emulation was separately shown
    # to differ materially from its own FP32 result on this trained model.
    graph_path = Path(control['graph_check'])
    graph = json.loads(graph_path.read_text())
    if (sha256(graph_path) != control['graph_check_sha256']
            or graph['implementation_sha256'] != sha256(Path(__file__).with_name('mac_mlx.py'))
            or graph['precision'] != 'float32' or graph['official_targets_advanced'] != 0
            or graph['comparison']['parameter_count'] != len(engine.params)
            or graph['comparison']['cosine'] < .99999 or graph['comparison']['relative_l2'] > .002):
        raise ValueError('Independent model derivative check differs')
    before = copy.deepcopy(reader.state_dict())
    rows = reader.next_batch(1)
    reader.load_state_dict(before)
    if hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest() != result['rows_sha256']:
        raise ValueError('CPU oracle row identity differs')
    count = int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
    value, gradients = engine.gradient(engine.params, mx.array(rows), engine.indices(rows))
    gradients = {n: g / count for n, g in gradients.items()}
    norm = mx.sqrt(sum(mx.sum(gradients[n] ** 2) for n in sorted(gradients)))
    gradients = {n: g * mx.minimum(1., 1. / (norm + 1e-6)) for n, g in gradients.items()}
    mx.eval(value, gradients, norm)
    comparison = full_gradient_check(gradients, reference=control['gradients'])
    delta = abs(float(value) / count - result['backends']['cpu']['loss'])
    passed = comparison['cosine'] > .999 and comparison['relative_l2'] < .05 and delta < .005
    write_json(output, {'passed': passed, 'comparison': comparison, 'absolute_loss_difference': delta,
        'cpu_results_sha256': sha256(results_path), 'cosine_minimum': .999, 'relative_l2_maximum': .05,
        'loss_difference_maximum': .005, 'official_targets_advanced': 0,
        'compute_precision': 'bfloat16', 'oracle_precision': 'float32', 'sequence_length': 8192,
        'graph_check_sha256': sha256(graph_path)})
    if not passed:
        raise RuntimeError('Full8K BF16 derivative check failed; preserve evidence')


def qualify(root, plan, old, plan_sha, report):
    root.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    report({'phase': 'backend-checks', 'training_started': False})
    with (root / 'backend-checks.log').open('x') as stream:
        checked = subprocess.run([sys.executable, '-m', 'pytest', '-q',
            str(source / 'tests/test_mac_mlx.py'), str(source / 'tests/test_mac_mlx_run.py'),
            str(source / 'tests/test_mac_speed.py')], cwd=source,
            stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    if checked.returncode:
        raise RuntimeError('MLX backend/checkpoint/refusal checks failed; preserved log')
    engine, state = import_mac_checkpoint(plan)
    stager = base.stager_for(old, 0); reader = stager.reader(state['data'])
    validator = Validation(plan['validation'], stager); validator.progress = report
    report({'phase': 'cpu-oracle-check', 'training_started': False})
    check_cpu_oracle(engine, reader, plan, root / 'cpu-oracle-check.json', report)
    baseline = validator(engine)
    base.check_losses(baseline, plan['validation']['final_reference'], plan['backend_reference_tolerance'], absolute=True)
    write_json(root / 'baseline.json', baseline)
    rows, metrics, saved = [], [], None
    for index in range(2):
        measured = advance(engine, reader, state, lambda v: report({'phase': 'qualification-update', 'update': index + 1, **v}))
        rows.append(measured['rows_sha256']); metrics.append(measured)
        if index == 0:
            saved = publish(root / 'restart-fixture', engine, state, plan_sha, plan['parent'], kind='qualification-only')
    expected = signatures(engine, state)
    del engine; gc.collect(); mx.clear_cache()
    engine, restored = restore(saved, plan_sha, kind='qualification-only')
    reader.load_state_dict(restored['data'])
    replay = advance(engine, reader, restored, lambda v: report({'phase': 'qualification-replay', **v}))
    actual = signatures(engine, restored)
    comparison = {'model_exact': actual['model'] == expected['model'], 'adam_exact': actual['adam'] == expected['adam'],
                  'reader_exact': actual['reader'] == expected['reader'], 'rng_exact': actual['rng'] == expected['rng'],
                  'rows_exact': replay['rows_sha256'] == rows[1], 'expected': expected, 'actual': actual}
    write_json(root / 'restart-comparison.json', comparison)
    if not all(comparison[n] for n in ('model_exact', 'adam_exact', 'reader_exact', 'rng_exact', 'rows_exact')):
        raise RuntimeError('Native model/Adam/reader/RNG restart not exact')
    after = validator(engine)
    base.check_losses(after, {s: r['loss'] for s, r in baseline.items()}, plan['max_validation_increase'])
    receipt = {'schema': SCHEMA, 'kind': 'mlx8k-qualification', 'passed': True, 'plan_sha256': plan_sha,
               'hardware': hardware(), 'implementation': bindings(), 'sequence_length': 8192, 'microbatches': 16,
               'baseline': baseline, 'after': after, 'restart_comparison_sha256': sha256(root / 'restart-comparison.json'),
               'backend_checks_sha256': sha256(root / 'backend-checks.log'),
               'cpu_oracle_check_sha256': sha256(root / 'cpu-oracle-check.json'),
               'metrics': metrics, 'official_parent_tokens_advanced': 0, 'download_bytes_requested': stager.download_bytes}
    write_json(root / 'qualification.json', receipt)
    del engine; gc.collect(); mx.clear_cache()
    return receipt


def retain_two(root, plan_sha):
    folders = sorted(p for p in root.glob('step-*') if p.is_dir() and not p.name.endswith('.partial'))
    if len(folders) <= 2:
        return
    latest = json.loads((root / 'latest.json').read_text())
    if (latest['checkpoint'] != folders[-1].name
            or latest['manifest_sha256'] != sha256(folders[-1] / 'mlx_manifest.json')):
        raise ValueError('Newest complete checkpoint pointer differs; no retention')
    for path in folders[:-2]:
        if path.is_symlink(): raise ValueError('Refusing symlink retention candidate')
        manifest = json.loads((path / 'mlx_manifest.json').read_text())
        if (manifest['kind'] != 'mac-production' or manifest['plan_sha256'] != plan_sha
                or path.name != f"step-{manifest['step']:08d}"):
            raise ValueError('Refusing unowned/protected retention candidate')
        archive = root.parent / 'checkpoint-history'; archive.mkdir(exist_ok=True)
        write_json(archive / (path.name + '.json'), manifest)
        shutil.copyfile(path / 'state.json', archive / (path.name + '.state.json'))
        shutil.rmtree(path)


def train(root, plan, old, receipt, plan_sha, report, stopped, resume=None):
    if (receipt['schema'] != SCHEMA or not receipt['passed'] or receipt['plan_sha256'] != plan_sha
            or receipt['hardware'] != hardware() or receipt['implementation'] != bindings()):
        raise ValueError('Production requires current source/hardware-bound MLX qualification')
    root.mkdir(exist_ok=True)
    if resume:
        engine, state = restore(base.external(resume), plan_sha)
    else:
        if (root / 'checkpoints/latest.json').exists(): raise ValueError('Explicit resume required')
        engine, state = import_mac_checkpoint(plan)
    if state['config'] != plan['config'] or state['validation'] != plan['validation']:
        raise ValueError('Native restart recipe differs')
    if state['tokens'] >= state['config']['target_tokens']: raise ValueError('Existing data target complete')
    stager = base.stager_for(old, 0); reader = stager.reader(state['data'])
    reader.coverage(plan['config']['target_tokens'])
    validator = Validation(plan['validation'], stager); validator.progress = report
    if resume:
        initial = validator(engine)
        base.check_losses(initial, {s: r['loss'] for s,r in receipt['baseline'].items()}, plan['max_validation_increase'])
    else:
        initial = receipt['baseline']
    state['last_validation'] = {'step': state['step'], 'tokens': state['tokens'], 'sources': initial}
    from .mac_tracking import Tracker
    tracker = Tracker(root / 'tracking', plan_sha, {'backend': 'mlx', 'precision': 'bf16', 'config': plan['config'],
        'parent_manifest_sha256': plan['parent_manifest_sha256'], 'plan_sha256': plan_sha, 'hardware': hardware()})
    def event(value):
        report(value); tracker.log(value)
    last_saved = state['step'] if resume else None
    try:
        event({'phase': 'training-started', 'step': state['step'], 'tokens': state['tokens'],
               'wandb_id': tracker.identity['id'], 'wandb_mode': tracker.mode,
               'wandb_url': tracker.run.url})
        while state['tokens'] < plan['config']['target_tokens'] and not stopped():
            measured = advance(engine, reader, state, lambda v: report({'phase': 'training-microbatch', 'step': state['step'], **v}))
            event({'phase': 'training', 'step': state['step'], 'tokens': state['tokens'], **measured})
            final = state['tokens'] >= plan['config']['target_tokens']
            # Save the consistent update before a long validation. A changed
            # restart always validates again; no pending result is reused.
            if last_saved is None or state['step'] % plan['config']['save_every'] == 0 or final:
                path = publish(root / 'checkpoints', engine, state, plan_sha, plan['parent'])
                last_saved = state['step']; retain_two(root / 'checkpoints', plan_sha)
                event({'phase': 'checkpoint', 'step': state['step'], 'tokens': state['tokens'], 'checkpoint': str(path)})
            if state['step'] % plan['config']['eval_every'] == 0 or final:
                measured_validation = validator(engine)
                base.check_losses(measured_validation, {s:r['loss'] for s,r in receipt['baseline'].items()},
                                  plan['max_validation_increase'])
                state['last_validation'] = {'step': state['step'], 'tokens': state['tokens'], 'sources': measured_validation}
                event({'phase': 'validation', **state['last_validation']})
        state['phase'] = 'completed' if state['tokens'] >= plan['config']['target_tokens'] else 'paused'
        if last_saved != state['step']:
            publish(root / 'checkpoints', engine, state, plan_sha, plan['parent'])
            retain_two(root / 'checkpoints', plan_sha)
        event({'phase': state['phase'], 'step': state['step'], 'tokens': state['tokens']})
        tracker.finish()
    except BaseException:
        tracker.finish(exit_code=1); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    for n in ('base-plan', 'parent', 'output', 'cpu-oracle-control'): prepare.add_argument('--' + n, type=Path, required=True)
    for command in ('pipeline', 'train'):
        q = sub.add_parser(command)
        q.add_argument('--plan', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
        q.add_argument('--until-stopped', action='store_true', required=True)
        if command == 'train':
            q.add_argument('--qualification', type=Path, required=True); q.add_argument('--resume', type=Path, required=True)
    args = p.parse_args()
    if not mx.is_available(mx.gpu): raise RuntimeError('MLX GPU unavailable')
    mx.set_default_device(mx.gpu); torch.set_num_threads(2)
    if args.command == 'prepare':
        print(json.dumps(make_plan(args), indent=2)); return
    plan, old = load_plan(args.plan); plan_sha = sha256(args.plan)
    mx.set_memory_limit(plan['memory_limit_bytes']); mx.set_cache_limit(512 * 2**20)
    root = base.external(args.output)
    if args.command == 'pipeline': root.mkdir(exist_ok=False)
    stopped = False
    live = {'training_started': False}
    def stop(sig, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    def report(value):
        if value.get('phase') in ('training-started', 'training', 'training-microbatch'):
            live['training_started'] = True
        for key in ('wandb_id', 'wandb_mode', 'wandb_url'):
            if key in value: live[key] = value[key]
        item = {'unix': time.time(), 'pid': os.getpid(), 'backend': 'mlx', 'precision': 'bf16',
                'plan_sha256': plan_sha, **live, **value}
        write_json(root / 'worker-status.json', item)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n'); stream.flush()
        print(json.dumps(item), flush=True)
    try:
        with base.device_lock(Path(old['archive_cache']).parent):
            if args.command == 'pipeline':
                report({'phase': 'qualifying', 'training_started': False})
                receipt = qualify(root / 'qualification', plan, old, plan_sha, report)
                if stopped or (root / 'STOP').exists():
                    report({'phase': 'paused-after-qualification'}); return
                train(root / 'training', plan, old, receipt, plan_sha, report,
                      lambda: stopped or (root / 'STOP').exists())
            else:
                receipt = json.loads(args.qualification.read_text())
                train(root / 'training', plan, old, receipt, plan_sha, report,
                      lambda: stopped or (root / 'STOP').exists(), resume=args.resume)
    except BaseException as error:
        report({'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
