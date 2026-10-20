"""Install, prepare in parallel, optionally qualify, then launch fresh 10B training."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from microllama.common.resources import tokenization_budget
from scripts.cloud import run_vast as launcher
from scripts.cloud.hardware import profiles

REPO = launcher.REPO
TARGET = 10_000_000_000


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def acquire_locks(root):
    locks = []
    try:
        for path in (REPO / '.runtime/experiments/gpu.lock', root / '.vast-launch.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locks
    except BaseException:
        for handle in locks:
            handle.close()
        raise RuntimeError('A setup/training launcher already owns this repository or data root') from None


def require_fresh(root, name):
    logs = root / 'logs' / name
    checkpoints = root / 'checkpoints' / name
    if any((logs / file).exists() for file in ('metrics.jsonl', 'status.json', 'wandb-id.json')) or any(checkpoints.glob('step-*')):
        raise RuntimeError(f'{name} already has training state; resume explicitly or choose a new --run-name')


def frozen_chunks(root, config, minimum):
    """Select whole published chunks, including enough fixed validation blocks."""
    selected, train, validation = [], 0, 0
    for path in sorted((root / 'tokens').glob('*/manifest.json')):
        if path.parent.name.endswith('.partial'):
            continue
        meta = read_json(path)
        selected.append(path.parent)
        train += meta['splits']['train']['tokens']
        validation += max(0, (meta['splits']['validation']['tokens'] - 1) // config['sequence_length'])
        if train >= minimum and validation >= config['eval_batches']:
            return selected
    return None


def freeze_subset(root, destination, chunks):
    """Hardlink immutable chunks into a separate complete inventory; never seal production."""
    from microllama.data_io.encoding import require_encoding_policy
    # Imported by the data interpreter, which owns the preparation dependencies.
    source = read_json(root / 'tokens/manifest.json')
    require_encoding_policy(source, root / 'tokens')
    (destination / 'tokens').mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / 'tokenizer', destination / 'tokenizer')
    totals = {'train': 0, 'validation': 0}
    for chunk in chunks:
        target = destination / 'tokens' / chunk.name
        target.mkdir()
        for file in chunk.iterdir():
            if not file.is_file() or file.is_symlink():
                raise RuntimeError(f'Unexpected chunk payload: {file}')
            os.link(file, target / file.name)
        meta = read_json(chunk / 'manifest.json')
        require_encoding_policy(meta, chunk)
        for split in totals:
            totals[split] += meta['splits'][split]['tokens']
    launcher.save_json(destination / 'tokens/manifest.json', {
        **source, 'complete': True, 'status': 'complete', 'shards': [chunk.name for chunk in chunks],
        'train_tokens': totals['train'], 'validation_tokens': totals['validation']})


def qualification_result(root, config, args):
    logs = root / 'logs' / config['run_name']
    status = read_json(logs / 'status.json') or {}
    if (status.get('phase') != 'stopped' or status.get('step') != args.qualification_updates or
            not status.get('qualification_passed') or status.get('world_size') != args.gpus or
            status.get('architecture') != ('engram' if args.engram else 'dense') or
            status.get('wandb_error') or status.get('error')):
        raise RuntimeError(f'Qualification did not pass: see {logs / "status.json"}')
    if not args.no_wandb and not status.get('wandb_url'):
        raise RuntimeError('Qualification W&B run was not established')
    if not math.isfinite(status.get('validation_loss', float('nan'))):
        raise RuntimeError('Qualification validation loss is not finite')
    metrics = [json.loads(line) for line in (logs / 'metrics.jsonl').read_text().splitlines()]
    if [entry['step'] for entry in metrics] != list(range(1, args.qualification_updates + 1)):
        raise RuntimeError('Qualification updates are incomplete')
    if any(not math.isfinite(entry[key]) for entry in metrics for key in ('loss', 'grad_norm')):
        raise RuntimeError('Qualification contains nonfinite losses or gradients')
    checkpoint = root / 'checkpoints' / config['run_name'] / f'step-{args.qualification_updates:08d}'
    if status.get('checkpoint') != str(checkpoint) or read_json(checkpoint / 'checkpoint_manifest.json')['step'] != args.qualification_updates:
        raise RuntimeError('Qualification checkpoint does not match the final update')
    return status, checkpoint


def training_spec(args, root, config, cap=None):
    path = root / 'logs' / config['run_name'] / 'launch-config.json'
    spec = SimpleNamespace(train_python=REPO / args.hardware['environment'] / 'bin/python',
        gpus=args.gpus, token_dir=root / 'tokens', checkpoint_dir=root / 'checkpoints',
        checkpoint_keep=args.checkpoint_keep, resume=None, continue_from=None, init_from=None,
        parent_manifest=None, max_steps=cap, engram=args.engram, no_wandb=args.no_wandb)
    return path, launcher.training_command(spec, path)


def run_workflow(args, locks):
    root, config = args.data_dir, args.training_config
    logs = root / 'logs/fast-setup'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'MICROLLAMA_DATA_ROOT': str(root), 'OMP_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false'}
    # Preserve an explicit shared cache; otherwise keep it outside virtualenvs.
    env.setdefault('HF_HOME', str(root.parent / '.hf_home'))
    children = []
    def report(phase, **fields):
        launcher.save_json(logs / 'status.json', {'phase': phase, 'pid': os.getpid(),
            'updated_unix': time.time(), 'run_name': config['run_name'], **fields})
        print(json.dumps({'phase': phase, **fields}), flush=True)
    def spawn(name, command, child_env=env, owner=False):
        with (logs / (name + '.log')).open('a') as log:
            child = subprocess.Popen([str(v) for v in command], cwd=REPO, env=child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=tuple(handle.fileno() for handle in locks) if owner else ())
        children.append((name, child))
        return child
    def check():
        for name, child in children:
            if child.poll() not in (None, 0):
                raise RuntimeError(f'{name} exited with {child.returncode}; see {logs / (name + ".log")}')
    def wait(child):
        while child.poll() is None:
            check()
            time.sleep(1)
        check()
    handed_off = False
    try:
        report('installing')
        if not args.skip_setup:
            interpreters = launcher.setup_interpreters(env)
            # Separate processes keep logs distinct and let failure cancel both installs.
            code = 'from scripts.cloud.run_vast import setup_environment; import os; '
            data_setup = spawn('data-environment', [sys.executable, '-c', code +
                f'setup_environment(dict(os.environ), {interpreters!r})'])
            training_setup = spawn('training-environment', [sys.executable, '-c', code +
                f'setup_environment(dict(os.environ), {interpreters!r}, {args.hardware!r})'])
            wait(data_setup)
        else:
            training_setup = None
        data_python = REPO / '.venv/bin/python'
        train_python = REPO / args.hardware['environment'] / 'bin/python'
        wait(spawn('tokenizer', [data_python, '-m', 'microllama.data_io.tokenizer']))
        summary = read_json(root / 'tokens/manifest.json')
        data_children = []
        if not summary or not summary.get('complete') or summary.get('status') != 'complete':
            wait(spawn('resolve-source', [data_python, '-m', 'microllama.data_io.download', '--dry-run']))
            source = read_json(root / 'raw/fineweb-edu-10bt/manifest.json')
            budget = tokenization_budget(len(source['files']), args.workers, args.threads)
            data_children.append(spawn('download', [data_python, '-m', 'microllama.data_io.download',
                '--workers', str(min(8, budget['cpus']))]))
            data_children.append(spawn('tokenize', [data_python, '-m', 'microllama.data_io.parallel_prepare',
                '--root', root, '--workers', str(budget['workers']), '--threads', str(budget['threads'])]))
            report('preparing', cpu_budget=budget)
        else:
            report('reusing-complete-corpus')
        qualified = args.skip_qualification
        qualifier, qualified_status, qualified_config = None, None, None
        qualified_root = root / '.qualification' / config['run_name']
        preflight_done = False
        while True:
            check()
            training_ready = training_setup is None or training_setup.poll() == 0
            if training_ready and not preflight_done:
                wait(spawn('cuda-preflight', [train_python, '-c',
                    f'from scripts.cloud.hardware import cuda_preflight; cuda_preflight({args.gpus}, {args.hardware_profile!r})']))
                preflight_done = True
            if preflight_done and not qualified and qualifier is None:
                chunks = frozen_chunks(root, config, args.qualification_tokens)
                if chunks:
                    if qualified_root.exists():
                        raise RuntimeError(f'Previous qualification exists at {qualified_root}; choose a new --run-name')
                    # Bootstrap Python needs no tokenization dependencies. Freeze in the data environment.
                    wait(spawn('freeze-qualification', [data_python, '-c',
                        'from pathlib import Path; from scripts.cloud.fast_start import freeze_subset; '
                        f'freeze_subset(Path({str(root)!r}), Path({str(qualified_root)!r}), '
                        f'[Path(p) for p in {[str(p) for p in chunks]!r}])']))
                    qualified_config = {**config, 'run_name': config['run_name'] + '-qualification',
                        'target_tokens': config['warmup_tokens'] + args.qualification_updates * config['sequence_length'] *
                            config['micro_batch_size'] * config['gradient_accumulation_steps'] + 1}
                    path, command = training_spec(args, qualified_root, qualified_config, args.qualification_updates)
                    launcher.save_json(path, qualified_config)
                    qualifier = spawn('qualification', command, {**env, 'MICROLLAMA_DATA_ROOT': str(qualified_root)})
                    report('qualifying', qualification_root=str(qualified_root))
            if qualifier is not None and qualifier.poll() == 0 and not qualified:
                qualified_status, checkpoint = qualification_result(qualified_root, qualified_config, args)
                wait(spawn('verify-qualification', [train_python, '-c',
                    'from pathlib import Path; from microllama.training.distributed import verify_checkpoint; '
                    f'print(verify_checkpoint(Path({str(checkpoint)!r})))']))
                qualified = True
                report('qualification-passed', checkpoint=str(checkpoint), qualification_state_reused=False)
            summary = read_json(root / 'tokens/manifest.json') or {}
            if summary.get('status') == 'failed':
                raise RuntimeError(f'Tokenization failed: {summary.get("error")}')
            if (preflight_done and qualified and summary.get('complete') and summary.get('status') == 'complete'
                    and all(child.poll() == 0 for child in data_children)):
                break
            if all(child.poll() == 0 for child in data_children) and data_children and not summary.get('complete'):
                raise RuntimeError('Preparation exited without a complete corpus')
            if not qualifier and not qualified and summary.get('complete') and not frozen_chunks(root, config, args.qualification_tokens):
                raise RuntimeError('Complete corpus is too small for the requested qualification/validation')
            time.sleep(2)
        require_fresh(root, config['run_name'])
        if shutil.disk_usage(root).free < launcher.checkpoint_free_bytes(keep=args.checkpoint_keep):
            raise RuntimeError('Insufficient free disk for retained checkpoints and publication reserve')
        path, command = training_spec(args, root, config)
        launcher.save_json(path, config)
        production = spawn('training', command, owner=True)
        receipt = {'pid': production.pid, 'run_name': config['run_name'], 'command': command,
            'config': str(path), 'log': str(logs / 'training.log'), 'hardware_profile': args.hardware_profile,
            'started_unix': time.time(), 'qualification_state_reused': False, 'initial_step': 0,
            'initial_tokens': 0, 'target_tokens': TARGET}
        launcher.save_json(root / 'logs/vast-launch.json', receipt)
        report('starting-training', **receipt)
        # Observe a real update and provenance before handing ownership to detached torchrun.
        deadline = time.monotonic() + 1800
        while True:
            check()
            if production.poll() == 0:
                raise RuntimeError('Production exited before a verified training update')
            status = read_json(path.parent / 'status.json') or {}
            if status.get('wandb_error') or status.get('phase') == 'failed':
                raise RuntimeError(f'Production failed: see {path.parent / "status.json"}')
            if status.get('step', 0) > 0:
                provenance = read_json(path.parent / 'provenance.json') or {}
                if ('lineage' not in provenance or provenance['lineage'] is not None or
                        status.get('world_size') != args.gpus or
                        status.get('architecture') != ('engram' if args.engram else 'dense') or
                        status['config']['target_tokens'] != TARGET or not math.isfinite(status.get('loss', float('nan')))):
                    raise RuntimeError('Production freshness/target/finite-loss checks failed')
                if not args.no_wandb and (not status.get('wandb_url') or
                        qualified_status and status['wandb_url'] == qualified_status.get('wandb_url')):
                    raise RuntimeError('Production must have its own new W&B run')
                receipt.update(verified_step=status['step'], wandb_url=status.get('wandb_url'))
                launcher.save_json(root / 'logs/vast-launch.json', receipt)
                report('training-started', **receipt)
                handed_off = True
                return
            if time.monotonic() > deadline:
                raise RuntimeError('Production did not complete its first update within 30 minutes')
            time.sleep(2)
    except BaseException as error:
        report('failed', error=str(error))
        raise
    finally:
        for name, child in children:
            if child.poll() is None and not (handed_off and name == 'training'):
                os.killpg(child.pid, signal.SIGTERM)
        for name, child in children:
            if handed_off and name == 'training':
                continue
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO / 'TrainData')
    parser.add_argument('--hardware-profile', choices=('4090-24', '4090-48', '5090-32'), default='4090-24')
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), default=4)
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument('--engram', dest='engram', action='store_true',
                              help='Use the revised Engram/AdamW recipe (default)')
    architecture.add_argument('--dense', dest='engram', action='store_false',
                              help='Use the dense AdamW baseline instead of Engram')
    parser.set_defaults(engram=True)
    parser.add_argument('--skip-qualification', action='store_true')
    parser.add_argument('--qualification-updates', type=int, default=40)
    parser.add_argument('--qualification-tokens', type=int, default=256_000_000,
                        help='Minimum early training tokens; validation readiness may require more')
    parser.add_argument('--workers', type=int, help='Override auto-sized source workers')
    parser.add_argument('--threads', type=int, help='Tokenizer threads per worker; auto-sized by default')
    parser.add_argument('--checkpoint-keep', type=int, default=4)
    parser.add_argument('--run-name')
    parser.add_argument('--skip-setup', action='store_true', help='Use already-installed pinned environments')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--foreground', action='store_true', help='Run orchestration in a terminal or Supervisor')
    parser.add_argument('--plan', action='store_true', help='Print stages/config without writes or network access')
    parser.add_argument('--lock-fds', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.hardware = profiles()[args.hardware_profile]
    args.data_dir = args.data_dir.expanduser().resolve()
    args.training_config = launcher.configuration(SimpleNamespace(resume=None, config=REPO / 'configs/train-8k-local.json',
        run_name=args.run_name, target_tokens=TARGET, micro_batch_size=None, hardware_profile=args.hardware_profile,
        custom_config=False, hardware=args.hardware, gpus=args.gpus, engram=args.engram))
    if args.qualification_updates < 3 or args.checkpoint_keep < 2:
        parser.error('Qualification needs at least three updates; retain at least two checkpoints')
    if args.qualification_tokens < 2 * (args.training_config['warmup_tokens'] + args.qualification_updates * 131072):
        parser.error('Qualification tokens must cover the unchanged warmup target and capped updates')
    tokenization_budget(14, args.workers, args.threads)
    return args


def main():
    args = parse_args()
    if args.plan:
        print(json.dumps({'config': args.training_config, 'gpus': args.gpus,
            'architecture': 'engram' if args.engram else 'dense',
            'cpu_budget': tokenization_budget(14, args.workers, args.threads),
            'stages': ['parallel environment installation', 'download + per-source parallel tokenization',
                'skip qualification' if args.skip_qualification else 'isolated early qualification + verified checkpoint',
                'complete corpus + fresh detached 10B training'], 'qualification_state_reused': False}, indent=2))
        return
    require_fresh(args.data_dir, args.training_config['run_name'])
    if args.lock_fds:
        locks = [os.fdopen(int(fd), 'a') for fd in args.lock_fds.split(',')]
    else:
        locks = acquire_locks(args.data_dir)
    try:
        if not args.foreground:
            logs = args.data_dir / 'logs/fast-setup'
            logs.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'scripts.cloud.fast_start', *sys.argv[1:], '--foreground',
                       '--lock-fds', ','.join(str(handle.fileno()) for handle in locks)]
            with (logs / 'workflow.log').open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(h.fileno() for h in locks))
            launcher.save_json(logs / 'workflow-launch.json', {'pid': child.pid, 'log': str(logs / 'workflow.log')})
            print(json.dumps({'detached_pid': child.pid, 'log': str(logs / 'workflow.log'),
                              'status': str(logs / 'status.json')}, indent=2))
            return
        def stop(signum, frame):
            raise KeyboardInterrupt('Workflow stopped; no automatic restart')
        signal.signal(signal.SIGTERM, stop)
        run_workflow(args, locks)
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
