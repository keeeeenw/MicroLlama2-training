import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microllama.common import resources
from scripts.cloud import fast_start as fast


def test_cpu_quota_and_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda pid: set(range(128)), raising=False)
    (tmp_path / 'cpu.max').write_text('6143999 100000')
    assert resources.available_cpus(tmp_path) == 61
    monkeypatch.setattr(resources, 'available_cpus', lambda: 61)
    assert resources.tokenization_budget(14) == {'cpus': 61, 'workers': 14, 'threads': 4}
    assert resources.tokenization_budget(1)['threads'] == 61
    with pytest.raises(ValueError, match='exceeds'):
        resources.tokenization_budget(14, 8, 8)


def test_plan_single_blackwell_is_fresh_and_same_batch(tmp_path):
    args = fast.parse_args(['--gpus', '1', '--hardware-profile', '5090-32', '--data-dir', str(tmp_path), '--plan'])
    path, command = fast.training_spec(args, tmp_path, args.training_config)
    assert '.venv-train-blackwell/bin/python' in command[0]
    assert '--nproc-per-node=1' in command
    assert args.engram is True and '--engram' in command
    assert '-engram-' in args.training_config['run_name']
    assert args.training_config['target_tokens'] == 10_000_000_000
    assert args.training_config['warmup_tokens'] == 100_000_000
    assert not set(command).intersection({'--resume', '--continue-from', '--init-from', '--max-steps'})
    assert not tmp_path.joinpath('logs').exists()


def test_locks_and_existing_history_prevent_duplicate_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, 'REPO', tmp_path / 'repo')
    root = tmp_path / 'data'
    locks = fast.acquire_locks(root)
    try:
        with pytest.raises(RuntimeError, match='already owns'):
            fast.acquire_locks(root)
    finally:
        for lock in locks:
            lock.close()
    path = root / 'logs/test/status.json'
    fast.launcher.save_json(path, {'phase': 'completed'})
    with pytest.raises(RuntimeError, match='already has'):
        fast.require_fresh(root, 'test')


def qualification_fixture(root, args):
    config = {'run_name': 'test-qualification'}
    logs = root / 'logs' / config['run_name']
    checkpoint = root / 'checkpoints' / config['run_name'] / 'step-00000040'
    status = dict(phase='stopped', step=40, qualification_passed=True, world_size=args.gpus,
                  architecture='engram' if args.engram else 'dense', validation_loss=9.5,
                  checkpoint=str(checkpoint), wandb_url='https://wandb.test/qualification')
    fast.launcher.save_json(logs / 'status.json', status)
    (logs / 'metrics.jsonl').write_text(''.join(json.dumps({'step': i, 'loss': 9.5, 'grad_norm': 1.2}) + '\n'
                                              for i in range(1, 41)))
    fast.launcher.save_json(checkpoint / 'checkpoint_manifest.json', {'step': 40})
    return config, logs, checkpoint


@pytest.mark.parametrize('failure', ['nonfinite', 'wandb', 'ranks', 'step', 'validation'])
def test_qualification_failure_blocks_production(tmp_path, failure):
    args = SimpleNamespace(gpus=4, qualification_updates=40, engram=True, no_wandb=False)
    config, logs, checkpoint = qualification_fixture(tmp_path, args)
    if failure == 'nonfinite':
        (logs / 'metrics.jsonl').write_text((logs / 'metrics.jsonl').read_text().replace('9.5', 'NaN', 1))
    else:
        status = fast.read_json(logs / 'status.json')
        status.update({'wandb': {'wandb_error': 'failed'}, 'ranks': {'world_size': 3},
                       'step': {'step': 39}, 'validation': {'validation_loss': float('nan')}}[failure])
        fast.launcher.save_json(logs / 'status.json', status)
    with pytest.raises(RuntimeError):
        fast.qualification_result(tmp_path, config, args)


@pytest.mark.parametrize('failure', [None, 'download', 'qualification', 'verify-qualification'])
def test_workflow_overlap_isolation_handoff_and_fail_closed(tmp_path, monkeypatch, failure):
    """Simulate external installers/trainers; execute the real orchestration decisions."""
    root = tmp_path / 'data'
    args = fast.parse_args(['--data-dir', str(root), '--run-name', 'test'])
    monkeypatch.setattr(fast, 'REPO', tmp_path / 'repo')
    config = args.training_config
    chunk = root / 'tokens/first-000000'
    fast.launcher.save_json(chunk / 'manifest.json', {'splits': {
        'train': {'tokens': 300_000_000}, 'validation': {'tokens': 2_000_000}}})
    fast.launcher.save_json(root / 'tokens/manifest.json', {'complete': False, 'status': 'preparing'})
    commands, flags, processes = {}, {}, {}
    monkeypatch.setattr(fast.launcher, 'setup_interpreters', lambda env: {'3.11': '/python311', '3.12': '/python312'})
    monkeypatch.setattr(fast.shutil, 'disk_usage', lambda root: SimpleNamespace(free=500 * 2**30))
    monkeypatch.setattr(fast.time, 'sleep', lambda seconds: None)

    class Process:
        def __init__(self, command, **kwargs):
            self.name = Path(kwargs['stdout'].name).stem
            self.pid = 999999
            self.returncode = 1 if self.name == failure else (None if self.name == 'training' else 0)
            commands[self.name], flags[self.name], processes[self.name] = command, kwargs, self
            if self.name == 'resolve-source':
                fast.launcher.save_json(root / 'raw/fineweb-edu-10bt/manifest.json', {'files': [{}] * 14})
            elif self.name == 'tokenize':
                fast.launcher.save_json(root / 'tokens/manifest.json', {'complete': True, 'status': 'complete'})
            elif self.name == 'qualification':
                qual_root = root / '.qualification/test'
                actual = fast.read_json(qual_root / 'logs/test-qualification/launch-config.json')
                assert actual['warmup_tokens'] == config['warmup_tokens']
                qualification_fixture(qual_root, args)
            elif self.name == 'training':
                path = root / 'logs/test'
                fast.launcher.save_json(path / 'status.json', {'phase': 'training', 'step': 1, 'loss': 10.0,
                    'config': config, 'world_size': args.gpus, 'architecture': 'engram',
                    'wandb_url': 'https://wandb.test/production'})
                fast.launcher.save_json(path / 'provenance.json', {'lineage': None})
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            return self.returncode
    monkeypatch.setattr(fast.subprocess, 'Popen', Process)
    lock = (tmp_path / 'owner.lock').open('a')
    try:
        if failure:
            with pytest.raises(RuntimeError, match='exited'):
                fast.run_workflow(args, [lock])
            assert 'training' not in commands
            assert fast.read_json(root / 'logs/fast-setup/status.json')['phase'] == 'failed'
            return
        fast.run_workflow(args, [lock])
        names = list(commands)
        assert names.index('training-environment') < names.index('tokenizer') < names.index('cuda-preflight')
        assert names.index('verify-qualification') < names.index('training')
        assert '--max-steps' in commands['qualification']
        assert not set(commands['training']).intersection({'--resume', '--init-from', '--continue-from', '--max-steps'})
        assert '/.qualification/' in commands['qualification'][commands['qualification'].index('--token-dir') + 1]
        assert flags['training']['start_new_session'] and flags['training']['pass_fds'] == (lock.fileno(),)
        receipt = fast.read_json(root / 'logs/vast-launch.json')
        assert receipt['initial_step'] == receipt['initial_tokens'] == 0
        assert receipt['qualification_state_reused'] is False and receipt['verified_step'] == 1
        assert fast.read_json(root / 'logs/fast-setup/status.json')['phase'] == 'training-started'
    finally:
        lock.close()


def test_frozen_subset_is_immutable_and_does_not_complete_production(tmp_path):
    from microllama.data_io.encoding import ENCODING_POLICY_VERSION
    root, frozen = tmp_path / 'data', tmp_path / 'qualification'
    chunk = root / 'tokens/first-000000'
    meta = {'encoding_policy_version': ENCODING_POLICY_VERSION,
            'splits': {'train': {'tokens': 300_000_000}, 'validation': {'tokens': 2_000_000}}}
    fast.launcher.save_json(chunk / 'manifest.json', meta)
    (chunk / 'train.bin').write_bytes(b'fixture')
    fast.launcher.save_json(root / 'tokens/manifest.json', {
        'complete': False, 'status': 'preparing', 'encoding_policy_version': ENCODING_POLICY_VERSION})
    fast.launcher.save_json(root / 'tokenizer/source_manifest.json', {'fixture': True})
    config = {'sequence_length': 8192, 'eval_batches': 128}
    selected = fast.frozen_chunks(root, config, 256_000_000)
    assert selected == [chunk]
    fast.freeze_subset(root, frozen, selected)
    assert (chunk / 'train.bin').stat().st_ino == (frozen / 'tokens' / chunk.name / 'train.bin').stat().st_ino
    assert fast.read_json(root / 'tokens/manifest.json')['complete'] is False
    assert fast.read_json(frozen / 'tokens/manifest.json')['complete'] is True
    fast.launcher.save_json(root / 'tokens/second-000000/manifest.json', meta)
    assert fast.read_json(frozen / 'tokens/manifest.json')['shards'] == [chunk.name]


def test_detached_workflow_and_trainer_keep_locks_after_parent_exits(tmp_path):
    """Exercise real process/session/FD handoff with a tiny CPU-only trainer stand-in."""
    import os
    import shutil
    import signal
    import subprocess
    import sys
    import time
    repo = tmp_path / 'checkout'
    source = Path(fast.__file__).resolve().parents[2]
    for relative in ('scripts', 'microllama/common', 'configs'):
        shutil.copytree(source / relative, repo / relative, ignore=shutil.ignore_patterns('__pycache__'))
    (repo / 'microllama/__init__.py').write_text('')
    # This CPU stand-in writes no model checkpoints. Keep the process/FD test
    # independent of host free space; real publication budgets have separate
    # tests. Override only the disposable checkout, never the actual launcher.
    launcher_copy = repo / 'scripts/cloud/run_vast.py'
    launcher_copy.write_text(launcher_copy.read_text()
                            + '\ncheckpoint_free_bytes = lambda *args, **kwargs: 1024\n')
    executable = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time
if "torch.distributed.run" in sys.argv:
    config_path = pathlib.Path(sys.argv[sys.argv.index("--config") + 1])
    config = json.loads(config_path.read_text())
    logs = config_path.parent
    (logs / "status.json").write_text(json.dumps({"phase": "training", "step": 1, "loss": 10., "config": config, "world_size": 1, "architecture": "engram" if "--engram" in sys.argv else "dense"}))
    (logs / "provenance.json").write_text(json.dumps({"lineage": None}))
    time.sleep(60)
'''
    for relative in ('.venv/bin/python', '.venv-train/bin/python'):
        path = repo / relative
        path.parent.mkdir(parents=True)
        path.write_text(executable)
        path.chmod(0o755)
    root = tmp_path / 'data'
    fast.launcher.save_json(root / 'tokens/manifest.json', {'complete': True, 'status': 'complete'})
    env = {**os.environ, 'PYTHONPATH': str(repo)}
    command = [sys.executable, '-m', 'scripts.cloud.fast_start', '--data-dir', str(root),
               '--run-name', 'detached-test', '--skip-setup', '--skip-qualification', '--no-wandb', '--gpus', '1']
    workflow_pid = training_pid = None
    try:
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        workflow_pid = json.loads(result.stdout)['detached_pid']
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            receipt = fast.read_json(root / 'logs/vast-launch.json')
            if receipt:
                training_pid = receipt['pid']
            status = fast.read_json(root / 'logs/fast-setup/status.json')
            if status and status['phase'] == 'training-started':
                break
            time.sleep(.1)
        else:
            pytest.fail((root / 'logs/fast-setup/workflow.log').read_text())
        assert os.getsid(training_pid) == training_pid
        # Allow the workflow to exit: the GPU lock must remain with detached training.
        time.sleep(.2)
        with (repo / '.runtime/experiments/gpu.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fast.fcntl.flock(lock, fast.fcntl.LOCK_EX | fast.fcntl.LOCK_NB)
        again = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=10)
        assert again.returncode != 0 and 'already has training state' in again.stderr
    finally:
        for pid in (workflow_pid, training_pid):
            if pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize('flags,engram', [([], True), (['--engram'], True), (['--dense'], False)])
def test_fast_start_architecture_default_and_override(tmp_path, flags, engram):
    args = fast.parse_args(['--data-dir', str(tmp_path), '--gpus', '1', *flags])
    assert args.engram is engram
    for root, config, cap in [(tmp_path, args.training_config, None),
                              (tmp_path / 'qualification', {**args.training_config, 'run_name': 'qualifier'}, 40)]:
        _, command = fast.training_spec(args, root, config, cap)
        assert ('--engram' in command) is engram
    with pytest.raises(SystemExit):
        fast.parse_args(['--engram', '--dense'])
