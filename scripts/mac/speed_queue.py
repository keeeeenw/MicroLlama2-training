"""One finite speed experiment after the production worker has stopped.

No recurring monitor, training continuation, failure retry or model promotion.
Every case is a separate process with a GPU lock and bounded execution time.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from microllama.training.mac import write_json, sha256
from microllama.training.mac_run import external, device_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pilot, root = external(args.pilot), external(args.output)
    root.mkdir(exist_ok=False)
    def status(value):
        write_json(root / 'status.json', {'unix': time.time(), 'pid': os.getpid(), **value})
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / 'microllama', root / 'code/microllama',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '._*'))
    (root / 'code/scripts/mac').mkdir(parents=True)
    shutil.copyfile(source / 'scripts/mac/qualify_mps.py', root / 'code/scripts/mac/qualify_mps.py')
    shutil.copyfile(source / 'scripts/mac/bench_speed.py', root / 'bench_speed.py')
    (root / 'preflight').mkdir()
    for name in ('test_mac_mlx.py', 'test_mac_training.py'):
        shutil.copyfile(source / 'tests' / name, root / 'preflight' / name)
    launch = json.loads((pilot / 'launch.json').read_text())
    status({'phase': 'waiting-for-safe-pause', 'training_pid': launch['pid']})
    while True:
        try:
            os.kill(launch['pid'], 0)
        except ProcessLookupError:
            break
        if (root / 'STOP').exists():
            status({'phase': 'cancelled-before-benchmarks'}); return
        time.sleep(5)
    outcome = json.loads((pilot / 'run/worker-status.json').read_text())
    explicit_early_stop = pilot / 'speed-work-early-stop-confirmed-20261018.json'
    if outcome['phase'] not in ('paused', 'completed') and not explicit_early_stop.exists():
        status({'phase': 'blocked', 'reason': 'No verified safe pause; inspect original worker history'})
        return
    pointer = json.loads((pilot / 'run/training/checkpoints/latest.json').read_text())
    checkpoint = pilot / 'run/training/checkpoints' / pointer['checkpoint']
    if sha256(checkpoint / 'mac_manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Latest safe checkpoint manifest changed')
    if explicit_early_stop.exists():
        stopped = json.loads(explicit_early_stop.read_text())
        if (stopped['pid'] != launch['pid'] or not stopped['human_authorized_discard_unsaved_updates']
                or stopped['retained_manifest_sha256'] != pointer['manifest_sha256']):
            raise ValueError('Early-stop confirmation does not bind the retained checkpoint')
    plan = launch['command'][launch['command'].index('--plan') + 1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'code'), 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1'}
    status({'phase': 'mlx-gpu-parity-preflight', 'checkpoint': str(checkpoint)})
    with device_lock(Path(plan).parent):
        with (root / 'mlx-gpu-parity.log').open('x') as stream:
            try:
                preflight = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    str(root / 'preflight/test_mac_mlx.py'), '-k', 'gpu'], env=env,
                    cwd=root / 'code', stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            except subprocess.TimeoutExpired:
                status({'phase': 'blocked', 'reason': 'GPU parity preflight timed out; no retry or full trials'})
                return
    if preflight.returncode:
        status({'phase': 'blocked', 'reason': 'MLX GPU parity gate failed; preserved log; no full trials'})
        return
    cases = [('torch-fp32-control', 'torch', 'float32', 256, 256),
             ('mlx-fp16-tiled', 'mlx', 'float16', 512, 1024),
             ('mlx-fp32-tiled', 'mlx', 'float32', 512, 1024),
             ('mlx-fp16-native-attention', 'mlx', 'float16', 0, 1024),
             ('torch-amp-fp16', 'torch', 'float16', 1024, 1024),
             ('torch-amp-fp16-full-attention', 'torch', 'float16', 8192, 1024),
             ('torch-fp32-larger-tiles', 'torch', 'float32', 1024, 1024)]
    results = {}
    for name, backend, precision, tile, chunk in cases:
        if (root / 'STOP').exists():
            status({'phase': 'cancelled-between-cases', 'completed_cases': list(results)}); return
        status({'phase': 'benchmarking', 'case': name, 'checkpoint': str(checkpoint)})
        command = [sys.executable, '-u', str(root / 'bench_speed.py'), '--checkpoint', str(checkpoint),
                   '--plan', plan, '--output', str(root / name), '--backend', backend,
                   '--precision', precision, '--tile', str(tile), '--chunk', str(chunk)]
        if backend == 'mlx':
            command.append('--compiled')
        with (root / (name + '.log')).open('x') as stream:
            try:
                process = subprocess.run(command, env=env, cwd=root / 'code', stdout=stream,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=900)
                outcome_path = root / name / 'result.json'
                results[name] = json.loads(outcome_path.read_text()) if outcome_path.exists() else {
                    'completed': False, 'exit_code': process.returncode, 'reason': 'No result; preserved log'}
            except subprocess.TimeoutExpired:
                results[name] = {'completed': False, 'reason': 'Bounded diagnostic timed out; no retry'}
    baseline = results['torch-fp32-control']
    comparisons = {}
    if baseline.get('completed'):
        import numpy as np
        initial = baseline['measurements'][-1]
        reference = np.concatenate([np.asarray(initial['gradient_samples'][name]) for name in sorted(initial['gradient_samples'])])
        for name, value in results.items():
            if not value.get('completed'):
                continue
            measured = value['measurements'][-1]
            actual = np.concatenate([np.asarray(measured['gradient_samples'][key]) for key in sorted(initial['gradient_samples'])])
            cosine = float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual)))
            loss_delta = abs(measured['loss'] - initial['loss'])
            comparisons[name] = {'speedup_vs_control': value['valid_targets_per_second'] / baseline['valid_targets_per_second'],
                'valid_targets_per_second': value['valid_targets_per_second'], 'absolute_loss_difference': loss_delta,
                'sampled_gradient_cosine': cosine,
                'same_rows': value['rows_sha256'] == baseline['rows_sha256'],
                'experimental_numerical_screen_passed': value['rows_sha256'] == baseline['rows_sha256'] and
                    loss_delta < (.02 if value['precision'] == 'float16' else .0001) and cosine > .999,
                'production_qualified': False}
    write_json(root / 'summary.json', {'kind': 'finite-speed-comparison', 'official_targets_advanced': 0,
        'comparisons': comparisons,
        'training_resumed': False, 'checkpoint': str(checkpoint), 'results': results})
    status({'phase': 'completed', 'training_resumed': False,
            'successful_cases': [name for name, value in results.items() if value.get('completed')]})


if __name__ == '__main__':
    main()
