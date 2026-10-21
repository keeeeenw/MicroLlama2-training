"""Start one detached, source-frozen Mac worker. No restart or target extension."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from microllama.training.mac import sha256, write_json
from microllama.training.mac_run import external, load_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--launch-root', type=Path, required=True)
    parser.add_argument('--max-download-bytes', type=int, required=True)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--max-seconds', type=int)
    parser.add_argument('--until-stopped', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.max_download_bytes < 0:
        raise ValueError('Nonnegative transfer allowance required')
    if args.until_stopped:
        if args.max_updates is not None or args.max_seconds is not None:
            raise ValueError('Choose until-stopped or explicit bounds, not both')
    elif args.max_updates is None or args.max_seconds is None or min(args.max_updates, args.max_seconds) < 1:
        raise ValueError('Explicit positive bounds or --until-stopped required')
    root = external(args.launch_root)
    root.mkdir(parents=True, exist_ok=False)
    code = root / 'code'
    repository = Path(__file__).resolve().parents[2]
    shutil.copytree(repository / 'microllama', code / 'microllama',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    (code / 'scripts/mac').mkdir(parents=True)
    shutil.copyfile(repository / 'scripts/mac/qualify_mps.py', code / 'scripts/mac/qualify_mps.py')
    for name, expected in plan['implementation'].items():
        if sha256(code / 'microllama' / name) != expected:
            raise RuntimeError('Source snapshot does not match the reviewed plan')
    temporary = root / 'tmp'
    temporary.mkdir()
    env = dict(os.environ)
    for name in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_RUN_GROUP'):
        env.pop(name, None)
    env.update(PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        HF_HUB_CACHE=str(root / 'hf-cache'), HF_XET_CACHE=str(Path(plan['archive_cache']) / 'xet'),
        HF_XET_CHUNK_CACHE_SIZE_BYTES='0', HF_HUB_DISABLE_TELEMETRY='1', TMPDIR=str(temporary),
        WANDB_SILENT='true')
    cmd = ['/usr/bin/caffeinate', '-i', '-s', sys.executable, '-u', '-m',
        'microllama.training.mac_run', 'pipeline', '--plan', str(args.plan.resolve()),
        '--output', str(root / 'run'), '--max-download-bytes', str(args.max_download_bytes)]
    cmd.extend(['--until-stopped'] if args.until_stopped else
               ['--max-updates', str(args.max_updates), '--max-seconds', str(args.max_seconds)])
    with (root / 'worker.log').open('xb') as log:
        child = subprocess.Popen(cmd, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': child.pid, 'unix': time.time(), 'command': cmd,
        'plan_sha256': sha256(args.plan), 'source_snapshot': str(code), 'autorestart': False,
        'max_download_bytes': args.max_download_bytes, 'max_updates': args.max_updates,
        'max_seconds': args.max_seconds, 'finite_target_tokens': plan['config']['target_tokens'],
        'execution_policy': 'until-user-stop-or-safety-gate' if args.until_stopped else 'bounded-pilot',
        'status': str(root / 'run/worker-status.json'), 'log': str(root / 'worker.log')}
    write_json(root / 'launch.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
