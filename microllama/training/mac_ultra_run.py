"""Source-frozen, larger-microbatch MLX continuation directly from Vast50B.

Existing Mac checkpoints are never accepted as initial parents. The tuning and
qualification updates are disposable; production imports the CUDA parent again.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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

from . import mac_mlx_run as native
from . import mac_run as base
from .mac import FINAL_MANIFEST, FINAL_STEP, FINAL_TOKENS, sha256, write_json
from .mac_mlx import adam_from_torch, adam_update
from ..data_io.growth_reader import load_catalog

SCHEMA = 'microllama-ultra-from-vast-v1'
GLOBAL_BATCH = 16
SOURCE_FILES = (*native.SOURCE_FILES, 'training/mac_ultra_run.py',
                '../tests/test_mac_ultra_run.py')


def bindings():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256(root / name) for name in SOURCE_FILES}


def batch_config(config, batch):
    if type(batch) is not int or batch not in (1, 2, 4, 8, 16):
        raise ValueError('Microbatch must divide the original global16 batch')
    return {**config, 'micro_batch_size': batch,
            'gradient_accumulation_steps': GLOBAL_BATCH // batch}


class BatchEngine(native.Engine):
    def __init__(self, *args, micro_batch_size=1, **kwargs):
        batch_config({}, micro_batch_size)
        super().__init__(*args, **kwargs)
        self.micro_batch_size = micro_batch_size

    def gradients(self, rows, report=lambda value: None):
        if rows.ndim != 2 or rows.shape[0] != GLOBAL_BATCH or rows.shape[1] < 2:
            raise ValueError('A complete ordered global16 batch is required')
        total, summed, accumulated = 0, 0., None
        started = time.monotonic()
        for first in range(0, GLOBAL_BATCH, self.micro_batch_size):
            batch = rows[first:first + self.micro_batch_size]
            valid = int(((batch[:, :-1] != 7) & (batch[:, :-1] != 0) & (batch[:, 1:] != 0)).sum())
            if valid < 1:
                raise ValueError('No valid targets in microbatch')
            value, gradients = self.gradient(self.params, mx.array(batch), self.indices(batch))
            accumulated = {n: gradients[n] + (0 if accumulated is None else accumulated[n])
                           for n in sorted(gradients)}
            mx.eval(value, accumulated)
            total += valid
            summed += float(value)
            if not math.isfinite(summed):
                raise RuntimeError('Nonfinite summed training loss')
            report({'microbatches_done': first // self.micro_batch_size + 1,
                    'microbatches_total': GLOBAL_BATCH // self.micro_batch_size,
                    'sequences_done': first + self.micro_batch_size,
                    'physical_batch_size': self.micro_batch_size,
                    'valid_targets_so_far': total, 'seconds': time.monotonic() - started})
            del value, gradients
        return summed / total, {n: accumulated[n] / total for n in sorted(accumulated)}, total

    def update(self, rows, report=lambda value: None):
        if rows.shape != (GLOBAL_BATCH, 8193):
            raise ValueError('Production requires original global16/full8K input')
        started = time.monotonic()
        mx.reset_peak_memory()
        value, gradients, total = self.gradients(rows, report)
        updated, moments, norm = adam_update(self.params, gradients, self.slots, self.policy)
        mx.eval(updated, moments, norm)
        if not math.isfinite(float(norm)):
            raise RuntimeError('Nonfinite gradient norm')
        self.params, self.slots = updated, moments
        elapsed = time.monotonic() - started
        return {'loss': value, 'valid_targets': total, 'grad_norm': float(norm),
                'seconds': elapsed, 'peak_bytes': mx.get_peak_memory(),
                'valid_targets_per_second': total / elapsed,
                'physical_batch_size': self.micro_batch_size,
                'gradient_accumulation_steps': GLOBAL_BATCH // self.micro_batch_size}


def verify_vast_parent(path):
    from .distributed import verify_checkpoint, validate_mixture_checkpoint
    path = base.external(path)
    if (path / 'mac_manifest.json').exists() or (path / 'mlx_manifest.json').exists():
        raise ValueError('Existing Mac checkpoints are forbidden as initial parents')
    if sha256(path / 'checkpoint_manifest.json') != FINAL_MANIFEST:
        raise ValueError('Only the original pinned Vast50B checkpoint is admitted')
    verify_checkpoint(path)
    saved = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved['step'] != FINAL_STEP or saved['tokens'] != FINAL_TOKENS:
        raise ValueError('Original CUDA counters differ')
    validate_mixture_checkpoint(saved, saved['config'], engram=True, world_size=4)
    if saved['data']['parent_tokens'] + sum(saved['data']['source_targets'].values()) != FINAL_TOKENS:
        raise ValueError('Original CUDA data counters do not reconcile')
    return saved


def import_vast(plan, batch=None):
    from .engram import EngramQwen3Config, EngramQwen3ForCausalLM
    from .optimizers import build_optimizer
    path = Path(plan['parent'])
    saved = verify_vast_parent(path)
    with torch.device('meta'):
        model = EngramQwen3ForCausalLM(EngramQwen3Config.from_pretrained(path, local_files_only=True))
    optimizer = build_optimizer(model, adam_lr=plan['config']['learning_rate'], fused=False,
                                memory_lr_multiplier=5, weight_decay=plan['config']['weight_decay'])
    slots, policy = adam_from_torch(model, optimizer, saved['optimizer'])
    for group in saved['optimizer']['param_groups']:
        memory = group.get('group_name') == 'memory_tables'
        if (group['lr'] != plan['config']['learning_rate'] * (5 if memory else 1)
                or tuple(group['betas']) != tuple(plan['config']['betas']) or group['eps'] != 1e-8):
            raise ValueError('Original Adam policy differs')
    names = sorted(n for n, _ in model.named_parameters())
    loaded = mx.load(str(path / 'model.safetensors'), stream=mx.cpu)
    params = {n: loaded[n].astype(mx.float32) for n in names}
    extras = {n: v for n, v in loaded.items() if n not in params and n != 'lm_head.weight'}
    if set(extras) - {'model.engram.canonical_ids'}:
        raise ValueError('Unexpected original model buffers')
    config = json.loads((path / 'config.json').read_text())
    batch = plan['config']['micro_batch_size'] if batch is None else batch
    state = {k: copy.deepcopy(saved[k]) for k in ('step', 'tokens', 'input_positions', 'data')}
    state.update(schema=native.SCHEMA, kind='mac-production',
                 config=batch_config(plan['config'], batch), validation=plan['validation'],
                 final_cuda_manifest_sha256=FINAL_MANIFEST, initial_parent_kind='original-vast-cuda',
                 original_cuda_rank_rng_preserved_in_parent=True, phase='ready')
    del loaded, saved, optimizer, model
    gc.collect()
    mx.random.seed(plan['config']['seed'])
    engine = BatchEngine(params, slots, policy, config, query_tile=plan['query_tile'],
                         chunk=plan['loss_chunk'], extras=extras, micro_batch_size=batch)
    mx.eval(engine.params, engine.slots)
    return engine, state


def restore_new_checkpoint(path, plan_sha, *, kind='qualification-only'):
    original, state = native.restore(path, plan_sha, kind=kind)
    if state.get('initial_parent_kind') != 'original-vast-cuda':
        raise ValueError('This launch must descend directly from original Vast50B')
    engine = BatchEngine(original.params, original.slots, original.policy, original.config,
                         query_tile=original.query_tile, chunk=original.chunk, extras=original.extras,
                         micro_batch_size=state['config']['micro_batch_size'])
    return engine, state


def advance(engine, reader, state, report=lambda value: None):
    before = copy.deepcopy(reader.state_dict())
    try:
        rows = reader.next_batch(GLOBAL_BATCH)
        metrics = engine.update(rows, report)
        after = reader.state_dict()
        tokens = after['parent_tokens'] + sum(after['source_targets'].values())
        if tokens - state['tokens'] != metrics['valid_targets']:
            raise RuntimeError('Reader/objective target counts differ')
    except BaseException:
        reader.load_state_dict(before)
        raise
    state.update(step=state['step'] + 1, tokens=tokens, data=after,
                 input_positions=state['input_positions'] + GLOBAL_BATCH * 8192, phase='training')
    metrics['rows_sha256'] = hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()
    return metrics


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan['schema'] != SCHEMA or plan['implementation'] != bindings()
            or plan['hardware'] != native.hardware() or plan['parent_manifest_sha256'] != FINAL_MANIFEST
            or plan['initial_parent_kind'] != 'original-vast-cuda'
            or plan['precision_environment'] != {'MLX_ENABLE_TF32': 'default'}
            or 'MLX_ENABLE_TF32' in os.environ
            or plan['max_download_bytes'] != 0 or plan['global_batch_size'] != GLOBAL_BATCH):
        raise ValueError('Plan source, hardware, original CUDA parent or batch binding differs')
    old = base.load_plan(plan['data_plan'])
    if sha256(plan['data_plan']) != plan['data_plan_sha256'] or plan['parent'] != old['parent']:
        raise ValueError('Data plan/original parent changed')
    expected = {**old['config'], 'precision': 'bf16', 'attention_query_tile': 512,
                'run_name': 'ultra-mlx-from-vast50b', 'eval_every': 250, 'save_every': 25}
    if plan['config'] != batch_config(expected, plan['config']['micro_batch_size']):
        raise ValueError('Preserved training recipe changed')
    if (plan['validation'] != old['validation'] or plan['query_tile'] != 512
            or plan['loss_chunk'] != 256 or plan['max_validation_increase'] != old['max_validation_increase']
            or plan['backend_reference_tolerance'] != .05
            or not 0 < plan['memory_limit_bytes'] <= int(plan['hardware']['memory_bytes'] * .75)):
        raise ValueError('Validation, numerical or memory policy changed')
    return plan, old


def prepare(args):
    old = base.load_plan(args.data_plan)
    saved = verify_vast_parent(old['parent'])
    target = old['config']['target_tokens']
    coverage = base.metadata_coverage(load_catalog(old['catalog']), saved['data'], target)
    if coverage['balanced_unconsumed_targets'] < target - FINAL_TOKENS + 131072:
        raise ValueError('Insufficient admitted data for complete-update target margin')
    config = {**old['config'], 'precision': 'bf16', 'attention_query_tile': 512,
              'run_name': 'ultra-mlx-from-vast50b', 'eval_every': 250, 'save_every': 25}
    hardware = native.hardware()
    memory_limit = min(64 * 2**30, int(hardware['metal']['max_recommended_working_set_size'] * .85))
    plan = {'schema': SCHEMA, 'initial_parent_kind': 'original-vast-cuda',
            'parent': old['parent'], 'parent_manifest_sha256': FINAL_MANIFEST,
            'parent_step': FINAL_STEP, 'parent_tokens': FINAL_TOKENS,
            'data_plan': str(args.data_plan.resolve()), 'data_plan_sha256': sha256(args.data_plan),
            'config': batch_config(config, 1), 'validation': old['validation'],
            'query_tile': 512, 'loss_chunk': 256, 'global_batch_size': GLOBAL_BATCH,
            'batch_candidates': [1, 2, 4, 8], 'implementation': bindings(), 'hardware': hardware,
            'memory_limit_bytes': memory_limit, 'max_download_bytes': 0,
            'precision_environment': {'MLX_ENABLE_TF32': 'default'},
            'max_validation_increase': old['max_validation_increase'], 'backend_reference_tolerance': .05,
            'execution_policy': 'qualify-then-automatically-train-until-stop-or-admitted-data-quality-limit',
            'new_cumulative_target': target, 'coverage': coverage,
            'precision_policy': 'BF16-compute-FP32-masters-Adam-global-gradient-accumulation'}
    out = base.external(args.output)
    if out.exists():
        raise ValueError('Never overwrite an existing plan')
    write_json(out, plan)
    print(json.dumps({'plan': str(out), 'sha256': sha256(out), 'hardware': hardware,
                      'target_tokens': target, 'memory_limit_bytes': memory_limit}), flush=True)


def probe(args, plan, old):
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    engine, state = import_vast(plan, args.batch)
    reader = base.stager_for(old, 0).reader(state['data'])
    rows = reader.next_batch(GLOBAL_BATCH)
    original_params, original_slots = engine.params, engine.slots
    measurements = []
    for trial in range(2):
        engine.params, engine.slots = original_params, original_slots
        mx.eval(engine.params, engine.slots)
        result = engine.update(rows, lambda v: print(json.dumps({'trial': trial, **v}), flush=True))
        measurements.append(result)
        write_json(root / 'progress.json', {'trial': trial, 'measurements': measurements})
    result = {'passed': True, 'batch': args.batch, 'measurements': measurements,
              'parent_manifest_sha256': FINAL_MANIFEST, 'official_targets_advanced': 0,
              'rows_sha256': hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()}
    write_json(root / 'result.json', result)


def cpu_reference(root, plan, old, report):
    from .engram import load_engram_model
    from .mac_attention import install_tiled_attention
    from .objective import chunked_loss, target_mask
    from .mac_gradient_check import full_gradient_check
    root.mkdir(exist_ok=False)
    saved = verify_vast_parent(plan['parent'])
    rows = base.stager_for(old, 0).reader(saved['data']).next_batch(1)
    del saved
    report({'phase': 'independent-cpu-reference', 'sequence_length': 8192})
    model = load_engram_model(plan['parent']).float().train()
    model.config.use_cache = False
    install_tiled_attention(model, plan['query_tile'])
    batch = torch.as_tensor(rows.astype(np.int64))
    x, y = batch[:, :-1], batch[:, 1:]
    mask = target_mask(x, y, 7, 0)
    count = int(mask.sum())
    started = time.monotonic()
    value = chunked_loss(model, x, y, chunk_size=256, loss_mask=mask, reduction='sum')
    value.backward()
    gradients = {n: p.grad / count for n, p in model.named_parameters()}
    evidence = full_gradient_check(gradients, export=root / 'gradients')
    result = {'kind': 'independent-fp32-cpu-diagnostic', 'official_targets_advanced': 0,
              'parent_manifest_sha256': FINAL_MANIFEST, 'sequence_length': 8192,
              'rows_sha256': hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest(),
              'loss': float(value.detach()) / count, 'seconds': time.monotonic() - started,
              'gradient_export': evidence}
    write_json(root / 'result.json', result)
    del model, gradients, value, batch, x, y, mask
    gc.collect()
    return result


def derivative_check(engine, reader, oracle_root, oracle, output):
    from .mac_gradient_check import full_gradient_check
    before = copy.deepcopy(reader.state_dict())
    rows = reader.next_batch(1)
    reader.load_state_dict(before)
    if hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest() != oracle['rows_sha256']:
        raise ValueError('Independent CPU and MLX row identities differ')
    valid = int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
    value, gradients = engine.gradient(engine.params, mx.array(rows), engine.indices(rows))
    gradients = {n: g / valid for n, g in gradients.items()}
    mx.eval(value, gradients)
    comparison = full_gradient_check(gradients, reference=oracle_root / 'gradients')
    delta = abs(float(value) / valid - oracle['loss'])
    passed = comparison['cosine'] > .999 and comparison['relative_l2'] < .05 and delta < .005
    write_json(output, {'passed': passed, 'comparison': comparison, 'absolute_loss_difference': delta,
                        'oracle_sha256': sha256(oracle_root / 'result.json'), 'sequence_length': 8192,
                        'cosine_minimum': .999, 'relative_l2_maximum': .05, 'loss_difference_maximum': .005})
    if not passed:
        raise RuntimeError('Full8K BF16 derivative check failed')


def qualify(root, plan, old, plan_sha, oracle_root, oracle, report):
    root.mkdir(exist_ok=False)
    engine, state = import_vast(plan)
    stager = base.stager_for(old, 0)
    reader = stager.reader(state['data'])
    # Independently reproduce the first complete CUDA-order global batch.
    from ..data_io.growth_reader import MixtureTokenBlocks
    before = copy.deepcopy(reader.state_dict())
    actual = reader.next_batch(GLOBAL_BATCH)
    reference = MixtureTokenBlocks(old['catalog'], 8192, seed=42)
    reference.load_state_dict(before)
    if not np.array_equal(actual, reference.next_batch(GLOBAL_BATCH)) or reader.state_dict() != reference.state_dict():
        raise RuntimeError('Independent CUDA-cursor/next-batch comparison failed')
    reader.load_state_dict(before)
    del reference, actual
    report({'phase': 'full8k-derivative-check'})
    derivative_check(engine, reader, oracle_root, oracle, root / 'derivative-check.json')
    validator = native.Validation(plan['validation'], stager)
    validator.progress = report
    baseline = validator(engine)
    base.check_losses(baseline, plan['validation']['final_reference'], .05, absolute=True)
    write_json(root / 'baseline.json', baseline)
    metrics = []
    for trial in range(2):
        metrics.append(advance(engine, reader, state,
            lambda v: report({'phase': 'qualification-update', 'trial': trial + 1, **v})))
        if trial == 0:
            fixture = native.publish(root / 'restart-fixture', engine, state, plan_sha,
                                     plan['parent'], kind='qualification-only')
    expected = native.signatures(engine, state)
    del engine
    gc.collect()
    mx.clear_cache()
    engine, restored = restore_new_checkpoint(fixture, plan_sha)
    reader.load_state_dict(restored['data'])
    replay = advance(engine, reader, restored, lambda v: report({'phase': 'qualification-replay', **v}))
    actual = native.signatures(engine, restored)
    comparison = {name + '_exact': expected[name] == actual[name] for name in expected}
    comparison['rows_exact'] = replay['rows_sha256'] == metrics[1]['rows_sha256']
    write_json(root / 'restart-comparison.json', {'passed': all(comparison.values()), **comparison,
                                                'expected': expected, 'actual': actual})
    if not all(comparison.values()):
        raise RuntimeError('Native model/Adam/reader/RNG replay was not exact')
    after = validator(engine)
    base.check_losses(after, {n: r['loss'] for n, r in baseline.items()}, plan['max_validation_increase'])
    receipt = {'schema': SCHEMA, 'passed': True, 'plan_sha256': plan_sha,
               'hardware': native.hardware(), 'implementation': bindings(),
               'parent_manifest_sha256': FINAL_MANIFEST, 'initial_parent_kind': 'original-vast-cuda',
               'sequence_length': 8192, 'physical_batch_size': engine.micro_batch_size,
               'global_batch_size': GLOBAL_BATCH, 'baseline': baseline, 'after': after,
               'metrics': metrics, 'official_targets_advanced': 0,
               'restart_comparison_sha256': sha256(root / 'restart-comparison.json'),
               'derivative_check_sha256': sha256(root / 'derivative-check.json'),
               'download_bytes_requested': stager.download_bytes}
    write_json(root / 'qualification.json', receipt)
    del engine
    gc.collect()
    mx.clear_cache()
    return receipt


def train(root, plan, old, receipt, plan_sha, report, stopped):
    if (not receipt['passed'] or receipt['plan_sha256'] != plan_sha
            or receipt['hardware'] != native.hardware() or receipt['implementation'] != bindings()
            or receipt['initial_parent_kind'] != 'original-vast-cuda'):
        raise ValueError('Production needs this exact hardware/source/parent qualification')
    root.mkdir(exist_ok=False)
    engine, state = import_vast(plan)  # Discard every tuning/qualification update.
    if state['step'] != FINAL_STEP or state['tokens'] != FINAL_TOKENS:
        raise RuntimeError('Production must start at original Vast counters')
    stager = base.stager_for(old, 0)
    reader = stager.reader(state['data'])
    reader.coverage(plan['config']['target_tokens'])
    validator = native.Validation(plan['validation'], stager)
    validator.progress = report
    from .mac_tracking import Tracker
    tracker = Tracker(root / 'tracking', plan_sha, {'backend': 'mlx', 'precision': 'bf16',
        'config': plan['config'], 'initial_parent_kind': 'original-vast-cuda',
        'parent_manifest_sha256': FINAL_MANIFEST, 'hardware': native.hardware()})
    tracker.run.name = 'm5-ultra-vast50b-' + tracker.identity['id']
    state['last_validation'] = {'step': FINAL_STEP, 'tokens': FINAL_TOKENS, 'sources': receipt['baseline']}
    def event(value):
        report(value)
        tracker.log(value)
    last_saved = None
    try:
        event({'phase': 'training-started', 'step': state['step'], 'tokens': state['tokens'],
               'wandb_id': tracker.identity['id'], 'wandb_mode': tracker.mode, 'wandb_url': tracker.run.url,
               'physical_batch_size': engine.micro_batch_size,
               'gradient_accumulation_steps': GLOBAL_BATCH // engine.micro_batch_size})
        while state['tokens'] < plan['config']['target_tokens'] and not stopped():
            metrics = advance(engine, reader, state, lambda v: report({'phase': 'training-microbatch',
                                                                      'step': state['step'], **v}))
            event({'phase': 'training', 'step': state['step'], 'tokens': state['tokens'], **metrics})
            final = state['tokens'] >= plan['config']['target_tokens']
            validating = state['step'] % plan['config']['eval_every'] == 0 or final
            if last_saved is None or state['step'] % plan['config']['save_every'] == 0 or validating or final:
                path = native.publish(root / 'checkpoints', engine, state, plan_sha, plan['parent'])
                last_saved = state['step']
                native.retain_two(root / 'checkpoints', plan_sha)
                event({'phase': 'checkpoint', 'step': state['step'], 'tokens': state['tokens'], 'checkpoint': str(path)})
            if validating:
                measured = validator(engine)
                base.check_losses(measured, {n: r['loss'] for n, r in receipt['baseline'].items()}, plan['max_validation_increase'])
                state['last_validation'] = {'step': state['step'], 'tokens': state['tokens'], 'sources': measured}
                event({'phase': 'validation', **state['last_validation']})
        state['phase'] = 'completed' if state['tokens'] >= plan['config']['target_tokens'] else 'paused'
        if last_saved != state['step']:
            native.publish(root / 'checkpoints', engine, state, plan_sha, plan['parent'])
            native.retain_two(root / 'checkpoints', plan_sha)
        event({'phase': state['phase'], 'step': state['step'], 'tokens': state['tokens']})
        tracker.finish()
    except BaseException:
        tracker.finish(exit_code=1)
        raise


def pipeline(args, plan, old):
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    stopping = False
    def stop(sig, frame):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def stopped():
        return stopping or (root / 'STOP').exists()
    def report(value):
        item = {'unix': time.time(), 'pid': os.getpid(), 'backend': 'mlx', 'precision': 'bf16',
                'parent_manifest_sha256': FINAL_MANIFEST, **value}
        write_json(root / 'worker-status.json', item)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')
        print(json.dumps(item), flush=True)
    try:
        with base.device_lock(Path(old['archive_cache']).parent):
            report({'phase': 'backend-tests'})
            source = Path(__file__).resolve().parents[2]
            with (root / 'backend-tests.log').open('x') as log:
                # The legacy FP32 reference tests require strict FP32 arithmetic.
                # This diagnostic process is separate from default-precision
                # BF16 production, which is checked below and on real 8K rows.
                subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    'tests/test_mac_mlx.py', 'tests/test_mac_mlx_run.py', 'tests/test_mac_speed.py',
                    'tests/test_mac_ultra_run.py'], cwd=source, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'MLX_ENABLE_TF32': '0'}, check=True, timeout=300)
            with (root / 'default-precision-tests.log').open('x') as log:
                subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                    'tests/test_mac_mlx.py', 'tests/test_mac_mlx_run.py', 'tests/test_mac_ultra_run.py',
                    '-k', 'not masked_forward_and_all_parameter_gradients'], cwd=source,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            candidates = []
            for batch in plan['batch_candidates']:
                if stopped():
                    report({'phase': 'paused-before-production'}); return
                folder = root / f'batch-{batch}'
                report({'phase': 'batch-benchmark', 'physical_batch_size': batch})
                with (root / f'batch-{batch}.log').open('x') as log:
                    try:
                        result = subprocess.run([sys.executable, '-B', '-u', '-m', 'microllama.training.mac_ultra_run', 'probe',
                            '--plan', str(args.plan), '--output', str(folder), '--batch', str(batch)],
                            stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                        if result.returncode == 0:
                            candidates.append(json.loads((folder / 'result.json').read_text()))
                        else:
                            write_json(root / f'batch-{batch}-failure.json', {'returncode': result.returncode})
                    except subprocess.TimeoutExpired:
                        write_json(root / f'batch-{batch}-failure.json', {'timeout_seconds': 1200})
            larger = [v for v in candidates if v['batch'] > 1]
            if not larger:
                raise RuntimeError('No larger microbatch passed; preserve benchmark evidence')
            if len({v['rows_sha256'] for v in candidates}) != 1:
                raise RuntimeError('Benchmark cases used different data rows')
            # Larger physical batches are tested and selected by measured update throughput.
            selected = max(larger, key=lambda v: v['measurements'][-1]['valid_targets_per_second'])
            plan = {**plan, 'config': batch_config(plan['config'], selected['batch'])}
            selected_path = root / 'selected-plan.json'
            write_json(selected_path, plan)
            plan, old = load_plan(selected_path)
            plan_sha = sha256(selected_path)
            write_json(root / 'batch-selection.json', {'selected_batch': selected['batch'],
                'candidates': candidates, 'plan_sha256': plan_sha, 'global_batch_size': GLOBAL_BATCH})
            report({'phase': 'batch-selected', 'physical_batch_size': selected['batch'],
                    'measured_valid_targets_per_second': selected['measurements'][-1]['valid_targets_per_second']})
            if stopped():
                report({'phase': 'paused-before-production'}); return
            oracle_root = root / 'cpu-reference'
            oracle = cpu_reference(oracle_root, plan, old, report)
            receipt = qualify(root / 'qualification', plan, old, plan_sha, oracle_root, oracle, report)
            if stopped():
                report({'phase': 'paused-after-qualification'}); return
            report({'phase': 'qualification-passed', 'plan_sha256': plan_sha})
            train(root / 'training', plan, old, receipt, plan_sha, report, stopped)
    except BaseException as error:
        report({'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise


def launch(args):
    plan, old = load_plan(args.plan)
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    code = root / 'code'
    shutil.copytree(source / 'microllama', code / 'microllama',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    for name, expected in plan['implementation'].items():
        target = code / 'microllama' / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'microllama' / name, target)
        if sha256(target) != expected:
            raise RuntimeError('Frozen source differs from plan')
    for name in ('test_mac_training.py',):
        target = code / 'tests' / name
        if not target.exists():
            shutil.copyfile(source / 'tests' / name, target)
    (root / 'tmp').mkdir()
    env = dict(os.environ)
    for name in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_RUN_GROUP', 'MLX_ENABLE_TF32'):
        env.pop(name, None)
    env.update(PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_CACHE=str(root / 'hf-cache'), HF_XET_CACHE=str(root / 'xet-cache'),
               TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
               TORCH_HOME=str(root / 'torch-cache'), WANDB_SILENT='true',
               GIT_PYTHON_GIT_EXECUTABLE=shutil.which('git') or 'git')
    command = ['/usr/bin/caffeinate', '-i', '-s', sys.executable, '-B', '-u', '-m',
               'microllama.training.mac_ultra_run', 'pipeline', '--plan', str(args.plan.resolve()),
               '--output', str(root / 'run'), '--until-stopped']
    with (root / 'worker.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'unix': time.time(), 'pid': process.pid, 'command': command,
               'source_snapshot': str(code), 'plan_sha256': sha256(args.plan),
               'parent_manifest_sha256': FINAL_MANIFEST, 'initial_parent_kind': 'original-vast-cuda',
               'auto_start_after_qualification': True, 'autorestart_on_failure': False,
               'status': str(root / 'run/worker-status.json'), 'target_tokens': plan['config']['target_tokens']}
    write_json(root / 'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for command in ('launch', 'pipeline', 'probe'):
        p = sub.add_parser(command)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if command == 'probe':
            p.add_argument('--batch', type=int, required=True)
        else:
            p.add_argument('--until-stopped', action='store_true', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    mx.set_default_device(mx.gpu)
    if args.command == 'prepare':
        prepare(args); return
    plan, old = load_plan(args.plan)
    mx.set_memory_limit(plan['memory_limit_bytes'])
    mx.set_cache_limit(512 * 2**20)
    if args.command == 'launch':
        launch(args)
    elif args.command == 'probe':
        probe(args, plan, old)
    else:
        pipeline(args, plan, old)


if __name__ == '__main__':
    main()
