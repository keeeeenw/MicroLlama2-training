"""Source-frozen stock MLX continuation after full original-Vast qualification."""
import argparse
import copy
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
import torch

from . import mac_fast_checkpoint as checkpoint, mac_mlx_run as native
from . import mac_ultra_run as ultra, mac_run as base
from .mac import FINAL_MANIFEST, FINAL_STEP, FINAL_TOKENS, sha256, write_json
from .mac_mlx_fast import FastEngine

OPTIONS = dict(micro_batch_size=1, chunk=1024, checkpoint=False,
               accumulation_window=1, fused_projections=False, gpu_embedding=False)
CACHE = 8 * 2**30
MEMORY = 64 * 2**30


def retain_two(root, plan_sha):
    """Retain owned checkpoints; macOS may unlink AppleDouble companions first."""
    folders = sorted(p for p in root.glob('step-*') if p.is_dir() and not p.name.endswith('.partial'))
    if len(folders) <= 2:
        return
    latest = json.loads((root / 'latest.json').read_text())
    if (latest['checkpoint'] != folders[-1].name
            or latest['manifest_sha256'] != sha256(folders[-1] / 'mlx_manifest.json')):
        raise ValueError('Newest checkpoint pointer differs; no retention')
    def missing_entry_only(function, path, error):
        if not isinstance(error, FileNotFoundError):
            raise error
    for path in folders[:-2]:
        if path.is_symlink():
            raise ValueError('Refusing symlink retention candidate')
        manifest = json.loads((path / 'mlx_manifest.json').read_text())
        if (manifest['kind'] != 'mac-production' or manifest['plan_sha256'] != plan_sha
                or path.name != f"step-{manifest['step']:08d}"):
            raise ValueError('Refusing unowned retention candidate')
        archive = root.parent / 'checkpoint-history'
        archive.mkdir(exist_ok=True)
        write_json(archive / (path.name + '.json'), manifest)
        shutil.copyfile(path / 'state.json', archive / (path.name + '.state.json'))
        shutil.rmtree(path, onexc=missing_entry_only)
        if path.exists():
            raise RuntimeError('Retention left an incomplete old checkpoint directory')


def resume_binding(root):
    """Bind a completed checkpoint to its exact frozen production execution."""
    root = Path(root).resolve()
    original_plan = root.parent / 'production-plan.json'
    plan_sha = sha256(original_plan)
    plan = json.loads(original_plan.read_text())
    if sha256(root.parent / 'code/microllama/training/mac_fast_run.py') != plan['launcher_sha256']:
        raise ValueError('Original frozen launcher differs')
    pointer = json.loads((root / 'checkpoints/latest.json').read_text())
    if Path(pointer['checkpoint']).name != pointer['checkpoint']:
        raise ValueError('Invalid checkpoint pointer')
    path = root / 'checkpoints' / pointer['checkpoint']
    manifest_sha = sha256(path / 'mlx_manifest.json')
    manifest = json.loads((path / 'mlx_manifest.json').read_text())
    state = json.loads((path / 'state.json').read_text())
    identity = json.loads((root / 'tracking/identity.json').read_text())
    if (pointer['manifest_sha256'] != manifest_sha or manifest['plan_sha256'] != plan_sha
            or manifest['kind'] != 'mac-production' or state['plan_sha256'] != plan_sha
            or state['initial_parent_kind'] != 'original-vast-cuda'
            or identity['plan_sha256'] != plan.get('resume', {}).get('tracking_identity', {}).get('plan_sha256', plan_sha)
            or identity['id'] == 'original-cuda-run'
            or state['fast_backend'] != {'backend': checkpoint.BACKEND,
                  'sources': checkpoint.bindings(), 'options': OPTIONS}):
        raise ValueError('Resume lineage/source/identity differs')
    return {'checkpoint': str(path), 'manifest_sha256': manifest_sha,
            'original_plan': str(original_plan), 'original_plan_sha256': plan_sha,
            'tracking_identity': identity, 'step': state['step'], 'tokens': state['tokens']}


def qualified(root, selected_sha):
    root = Path(root)
    contract = json.loads((root / 'qualification-plan.json').read_text())
    receipt = json.loads((root / 'qualification.json').read_text())
    derivative = json.loads((root / 'derivative-check.json').read_text())
    if (receipt.get('passed') is not True or receipt.get('full_128_block_validation') is not True
            or receipt.get('plan_sha256') != sha256(root / 'qualification-plan.json')
            or receipt.get('parent_manifest_sha256') != FINAL_MANIFEST
            or receipt.get('download_bytes_requested') != 0
            or not receipt.get('restart') or not all(receipt['restart'].values())
            or derivative.get('passed') is not True
            or contract.get('data_plan_sha256') != selected_sha
            or contract.get('initial_parent_kind') != 'original-vast-cuda'
            or contract.get('hardware') != native.hardware()
            or contract.get('blocks_per_source') != 128
            or contract.get('global_batch') != 16 or contract.get('sequence_length') != 8192
            or contract.get('cache_limit_bytes') != CACHE
            or contract.get('memory_limit_bytes') != MEMORY
            or contract.get('fast_backend') != {'backend': checkpoint.BACKEND,
                  'sources': checkpoint.bindings(), 'options': OPTIONS}):
        raise ValueError('Full qualification does not match current stock sources/options/hardware')
    if len(receipt['baseline']) != 5 or set(receipt['baseline']) != set(receipt['after']):
        raise ValueError('Qualification validation source sets differ')
    for source, before in receipt['baseline'].items():
        after = receipt['after'][source]
        if (any(v['blocks'] != 128 or v['valid_targets'] < 1_000_000
                or not math.isfinite(v['loss']) for v in (before, after))
                or before['ordered_rows_sha256'] != after['ordered_rows_sha256']
                or before['inventory_fingerprint'] != after['inventory_fingerprint']):
            raise ValueError('Incomplete or changed held-out validation')
    return receipt


def load(path):
    plan = json.loads(Path(path).read_text())
    if (plan['kind'] != 'qualified-stock-mlx-production-v1'
            or plan['launcher_sha256'] != sha256(Path(__file__))
            or plan['qualification_sha256'] != sha256(Path(plan['qualification']) / 'qualification.json')
            or plan['selected_plan_sha256'] != sha256(plan['selected_plan'])
            or plan['options'] != OPTIONS or plan['cache_limit_bytes'] != CACHE
            or plan['memory_limit_bytes'] != MEMORY):
        raise ValueError('Production plan/source binding differs')
    selected, old = ultra.load_plan(plan['selected_plan'])
    receipt = qualified(plan['qualification'], plan['selected_plan_sha256'])
    if plan.get('resume'):
        bound = plan['resume']
        if (sha256(bound['original_plan']) != bound['original_plan_sha256']
                or sha256(Path(bound['checkpoint']) / 'mlx_manifest.json') != bound['manifest_sha256']):
            raise ValueError('Resume checkpoint or original plan changed')
        original = json.loads(Path(bound['original_plan']).read_text())
        if any(original[n] != plan[n] for n in ('selected_plan_sha256', 'qualification_sha256',
                                               'options', 'cache_limit_bytes', 'memory_limit_bytes')):
            raise ValueError('Resume recipe differs from qualified original execution')
    return plan, selected, old, receipt


def worker(args):
    plan, selected, old, receipt = load(args.plan)
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    stop_requested = False
    def stop(sig, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def stopped():
        return stop_requested or (root / 'STOP').exists()
    def report(value):
        item = {'unix': time.time(), 'pid': os.getpid(), 'backend': 'mlx',
                'precision': 'bf16', 'initial_parent_kind': 'original-vast-cuda', **value}
        write_json(root / 'worker-status.json', item)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')
        print(json.dumps(item), flush=True)
    torch.set_num_threads(8)
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(MEMORY)
    mx.set_cache_limit(CACHE)
    tracker = None
    try:
        with base.device_lock(Path(old['archive_cache']).parent):
            resume = plan.get('resume')
            if resume:
                report({'phase': 'restore-verified-production-checkpoint', **resume})
                engine, state = checkpoint.restore(resume['checkpoint'],
                    resume['original_plan_sha256'], kind='mac-production')
                if (state['step'], state['tokens']) != (resume['step'], resume['tokens']):
                    raise ValueError('Restored counters differ')
                state['execution_recovery'] = {'plan_sha256': sha256(args.plan),
                    'restored_manifest_sha256': resume['manifest_sha256'],
                    'reason': 'AppleDouble checkpoint retention race; numerical graph unchanged'}
            else:
                report({'phase': 'import-original-vast'})
                original, state = ultra.import_vast(selected, 1)
                engine = FastEngine(original.params, original.slots, original.policy,
                                    original.config, extras=original.extras, **OPTIONS)
                del original
                if (state['step'], state['tokens']) != (FINAL_STEP, FINAL_TOKENS):
                    raise ValueError('Production must begin at original Vast50B counters')
            state['config'].update(micro_batch_size=1, gradient_accumulation_steps=16,
                                   loss_chunk_size=1024)
            if not resume:
                state['last_validation'] = {'step': FINAL_STEP, 'tokens': FINAL_TOKENS,
                                            'sources': receipt['baseline']}
            stager = base.stager_for(old, 0)
            reader = stager.reader(state['data'])
            reader.coverage(selected['config']['target_tokens'])
            validator = native.Validation(selected['validation'], stager)
            validator.progress = report
            if resume:
                measured = validator(engine)
                base.check_losses(measured, {n: r['loss'] for n, r in receipt['baseline'].items()},
                                  selected['max_validation_increase'])
                state['last_validation'] = {'step': state['step'], 'tokens': state['tokens'],
                                            'sources': measured}
            from .mac_tracking import Tracker
            tracking_sha = sha256(args.plan)
            if resume:
                (root / 'tracking').mkdir()
                write_json(root / 'tracking/identity.json', resume['tracking_identity'])
                tracking_sha = resume['tracking_identity']['plan_sha256']
            tracker = Tracker(root / 'tracking', tracking_sha, {
                'backend': 'mlx-stock-fast', 'precision': 'bf16', 'config': state['config'],
                'fast_backend': checkpoint.identity(engine), 'allocator_cache_bytes': CACHE,
                'initial_parent_kind': 'original-vast-cuda', 'parent_manifest_sha256': FINAL_MANIFEST,
                'hardware': native.hardware(), 'qualification': plan['qualification'],
                'execution_plan_sha256': sha256(args.plan)})
            tracker.run.name = 'm5-ultra-stock-vast50b-' + tracker.identity['id']
            def event(value):
                report(value)
                tracker.log(value)
            event({'phase': 'training-started', 'step': state['step'], 'tokens': state['tokens'],
                   'wandb_id': tracker.identity['id'], 'wandb_mode': tracker.mode,
                   'wandb_url': tracker.run.url, 'physical_batch_size': 1,
                   'gradient_accumulation_steps': 16, 'target_tokens': selected['config']['target_tokens']})
            event({'phase': 'validation', **state['last_validation']})
            last_saved = None
            while state['tokens'] < selected['config']['target_tokens'] and not stopped():
                metrics = ultra.advance(engine, reader, state, lambda v: report({
                    'phase': 'training-microbatch', 'step': state['step'], **v}))
                event({'phase': 'training', 'step': state['step'], 'tokens': state['tokens'], **metrics})
                updates = state['step'] - FINAL_STEP
                final = state['tokens'] >= selected['config']['target_tokens']
                validating = updates == 25 or updates % selected['config']['eval_every'] == 0 or final
                if last_saved is None or updates % selected['config']['save_every'] == 0 or validating:
                    path = checkpoint.publish(root / 'checkpoints', engine, state,
                        sha256(args.plan), selected['parent'], kind='mac-production')
                    last_saved = state['step']
                    retain_two(root / 'checkpoints', sha256(args.plan))
                    event({'phase': 'checkpoint', 'step': state['step'], 'tokens': state['tokens'],
                           'checkpoint': str(path)})
                if validating:
                    measured = validator(engine)
                    event({'phase': 'validation', 'step': state['step'], 'tokens': state['tokens'],
                           'sources': measured})
                    base.check_losses(measured, {n: r['loss'] for n, r in receipt['baseline'].items()},
                                      selected['max_validation_increase'])
                    state['last_validation'] = {'step': state['step'], 'tokens': state['tokens'],
                                                'sources': measured}
            state['phase'] = 'completed' if state['tokens'] >= selected['config']['target_tokens'] else 'paused'
            if last_saved != state['step']:
                checkpoint.publish(root / 'checkpoints', engine, state, sha256(args.plan),
                                   selected['parent'], kind='mac-production')
                retain_two(root / 'checkpoints', sha256(args.plan))
            event({'phase': state['phase'], 'step': state['step'], 'tokens': state['tokens']})
            tracker.finish()
    except BaseException as error:
        report({'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise


def launch(args):
    selected, old = ultra.load_plan(args.plan)
    qualified(args.qualification, sha256(args.plan))
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    code = root / 'code'
    shutil.copytree(source / 'microllama', code / 'microllama',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    for name, expected in checkpoint.bindings().items():
        target = code / 'microllama' / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'microllama' / name, target)
        if sha256(target) != expected:
            raise ValueError('Frozen source differs from qualification')
    plan = {'kind': 'qualified-stock-mlx-production-v1',
            'selected_plan': str(args.plan.resolve()), 'selected_plan_sha256': sha256(args.plan),
            'qualification': str(args.qualification.resolve()),
            'qualification_sha256': sha256(args.qualification / 'qualification.json'),
            'launcher_sha256': sha256(Path(__file__)), 'options': OPTIONS,
            'cache_limit_bytes': CACHE, 'memory_limit_bytes': MEMORY,
            'authorization': 'User requested automatic full launch after stock exploration and validation',
            'save_every_updates': 25, 'first_validation_update': 25, 'eval_every_updates': 250}
    if args.resume_run:
        plan['resume'] = resume_binding(args.resume_run)
    path = root / 'production-plan.json'
    write_json(path, plan)
    (root / 'tmp').mkdir()
    env = dict(os.environ)
    for name in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_RUN_GROUP', 'MLX_ENABLE_TF32'):
        env.pop(name, None)
    env.update(PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_SILENT='true',
               HF_HUB_CACHE=str(root / 'hf-cache'), HF_XET_CACHE=str(root / 'xet-cache'),
               TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
               TORCH_HOME=str(root / 'torch-cache'),
               GIT_PYTHON_GIT_EXECUTABLE=shutil.which('git') or 'git')
    # Pass the authorized login to the detached SDK service explicitly. The
    # restored netrc authenticates Python requests but service discovery may
    # differ when launched with external-drive cache directories. Never persist
    # this environment in the launch receipt or source snapshot.
    from wandb.sdk.lib import apikey
    key = apikey.api_key()
    if key:
        env['WANDB_API_KEY'] = key
    command = ['/usr/bin/caffeinate', '-i', '-s', sys.executable, '-B', '-u', '-m',
               'microllama.training.mac_fast_run', 'worker', '--plan', str(path),
               '--output', str(root / 'run')]
    with (root / 'worker.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'unix': time.time(), 'pid': process.pid, 'command': command,
               'source_snapshot': str(code), 'plan_sha256': sha256(path),
               'initial_parent_kind': 'original-vast-cuda', 'parent_manifest_sha256': FINAL_MANIFEST,
               'target_tokens': selected['config']['target_tokens'], 'autorestart_on_failure': False,
               'status': str(root / 'run/worker-status.json')}
    write_json(root / 'launch.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['launch', 'worker'])
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--qualification', type=Path)
    p.add_argument('--resume-run', type=Path)
    args = p.parse_args()
    if args.command == 'launch':
        if args.qualification is None:
            p.error('--qualification is required for launch')
        launch(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
