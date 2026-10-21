import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from microllama.training.mac_tracking import identity, OLD_CUDA_RUN
from microllama.training.mac_run import pipeline


def test_mac_tracking_identity_is_new_and_resumes_only_its_own_plan(tmp_path):
    first = identity(tmp_path, 'plan-a', mode='offline')
    assert first['id'] != OLD_CUDA_RUN
    assert identity(tmp_path, 'plan-a', mode='offline') == first
    with pytest.raises(ValueError, match='identity or plan differs'):
        identity(tmp_path, 'plan-b', mode='offline')
    first['id'] = OLD_CUDA_RUN
    (tmp_path / 'identity.json').write_text(json.dumps(first))
    with pytest.raises(ValueError, match='never reuse'):
        identity(tmp_path, 'plan-a', mode='offline')


def test_pipeline_qualifies_before_training_and_shares_one_transfer_cap(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('microllama.training.mac_run.external', lambda path: Path(path))
    def qualify(args, plan, plan_sha):
        calls.append(('qualify', args.max_download_bytes))
        args.output.mkdir()
        return {'passed': True, 'download_bytes_requested': 80}
    def train(args, plan, plan_sha):
        calls.append(('train', args.max_download_bytes))
        return {'phase': 'paused', 'updates': 2}
    monkeypatch.setattr('microllama.training.mac_run.qualification', qualify)
    monkeypatch.setattr('microllama.training.mac_run.train', train)
    root = tmp_path / 'worker'
    pipeline(SimpleNamespace(output=root, max_download_bytes=100, max_seconds=60, max_updates=2), {}, 'sha')
    assert calls == [('qualify', 100), ('train', 20)]
    assert json.loads((root / 'worker-status.json').read_text())['training_started'] is True


def test_failed_qualification_never_starts_training(tmp_path, monkeypatch):
    monkeypatch.setattr('microllama.training.mac_run.external', lambda path: Path(path))
    def fail(*args):
        raise RuntimeError('backend qualification failed')
    monkeypatch.setattr('microllama.training.mac_run.qualification', fail)
    monkeypatch.setattr('microllama.training.mac_run.train', lambda *args: pytest.fail('Must not train'))
    root = tmp_path / 'worker'
    with pytest.raises(RuntimeError, match='backend qualification'):
        pipeline(SimpleNamespace(output=root, max_download_bytes=100, max_seconds=60, max_updates=2), {}, 'sha')
    assert json.loads((root / 'worker-status.json').read_text())['phase'] == 'failed'
