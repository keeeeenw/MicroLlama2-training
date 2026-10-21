"""Production admission refuses stale sources, incomplete validation and failed replay."""
import copy
import json

import pytest

pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training import mac_fast_run as run
from microllama.training.mac import FINAL_MANIFEST, sha256


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run.native, 'hardware', lambda: {'chip': 'test'})
    monkeypatch.setattr(run.checkpoint, 'bindings', lambda: {'source': 'a'*64})
    contract = {'data_plan_sha256': 'b'*64, 'initial_parent_kind': 'original-vast-cuda',
        'hardware': {'chip': 'test'}, 'blocks_per_source': 128, 'global_batch': 16,
        'sequence_length': 8192, 'cache_limit_bytes': run.CACHE, 'memory_limit_bytes': run.MEMORY,
        'fast_backend': {'backend': run.checkpoint.BACKEND,
                         'sources': run.checkpoint.bindings(), 'options': run.OPTIONS}}
    (tmp_path/'qualification-plan.json').write_text(json.dumps(contract))
    metrics = {str(i): {'loss': 2., 'blocks': 128, 'valid_targets': 1_047_000,
                       'ordered_rows_sha256': 'c'*64, 'inventory_fingerprint': 'd'*64}
               for i in range(5)}
    receipt = {'passed': True, 'full_128_block_validation': True,
        'plan_sha256': sha256(tmp_path/'qualification-plan.json'),
        'parent_manifest_sha256': FINAL_MANIFEST, 'download_bytes_requested': 0,
        'restart': {'model_exact': True, 'adam_exact': True, 'rows_exact': True},
        'baseline': metrics, 'after': copy.deepcopy(metrics)}
    (tmp_path/'qualification.json').write_text(json.dumps(receipt))
    (tmp_path/'derivative-check.json').write_text(json.dumps({'passed': True}))
    return receipt


def test_admits_exact_full_qualification(tmp_path, monkeypatch):
    receipt = fixture(tmp_path, monkeypatch)
    assert run.qualified(tmp_path, 'b'*64) == receipt
    monkeypatch.setattr(run.checkpoint, 'bindings', lambda: {'source': 'e'*64})
    with pytest.raises(ValueError, match='qualification'):
        run.qualified(tmp_path, 'b'*64)


@pytest.mark.parametrize('failure', ['replay', 'short-validation', 'nan', 'rows', 'parent'])
def test_refuses_invalid_qualification(tmp_path, monkeypatch, failure):
    receipt = fixture(tmp_path, monkeypatch)
    if failure == 'replay': receipt['restart']['adam_exact'] = False
    if failure == 'short-validation': receipt['after']['0']['blocks'] = 8
    if failure == 'nan': receipt['after']['0']['loss'] = float('nan')
    if failure == 'rows': receipt['after']['0']['ordered_rows_sha256'] = 'wrong'
    if failure == 'parent': receipt['parent_manifest_sha256'] = 'wrong'
    (tmp_path/'qualification.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError): run.qualified(tmp_path, 'b'*64)


def retention_fixture(tmp_path):
    root = tmp_path / 'checkpoints'; root.mkdir()
    for step in (1, 25, 50):
        path = root / f'step-{step:08d}'; path.mkdir()
        (path/'state.json').write_text('{}')
        (path/'model').write_text('payload')
        (path/'._model').write_text('companion')
        (path/'mlx_manifest.json').write_text(json.dumps({
            'kind':'mac-production', 'plan_sha256':'f'*64, 'step':step}))
    newest = root/'step-00000050'
    (root/'latest.json').write_text(json.dumps({'checkpoint':newest.name,
        'manifest_sha256':sha256(newest/'mlx_manifest.json')}))
    return root


def test_retention_tolerates_disappearing_appledouble_companion(tmp_path, monkeypatch):
    root = retention_fixture(tmp_path)
    unlink = run.os.unlink
    def apple_double_unlink(path, *, dir_fd=None):
        if str(path) == '._model':
            # Model macOS removing the companion before rmtree's cached entry.
            unlink(path, dir_fd=dir_fd)
        return unlink(path, dir_fd=dir_fd)
    monkeypatch.setattr(run.shutil.os, 'unlink', apple_double_unlink)
    run.retain_two(root, 'f'*64)
    assert not (root/'step-00000001').exists()
    assert (root/'step-00000025/model').read_text() == 'payload'
    assert (root/'step-00000050/model').read_text() == 'payload'
    assert (tmp_path/'checkpoint-history/step-00000001.state.json').exists()


def test_retention_does_not_hide_permission_failure(tmp_path, monkeypatch):
    root = retention_fixture(tmp_path)
    def denied(path, *, dir_fd=None):
        raise PermissionError('denied')
    with monkeypatch.context() as patch:
        patch.setattr(run.shutil.os, 'unlink', denied)
        with pytest.raises(PermissionError): run.retain_two(root, 'f'*64)
    assert (root/'step-00000050/model').exists()


def test_retention_refuses_unowned_checkpoint(tmp_path):
    root = retention_fixture(tmp_path)
    with pytest.raises(ValueError, match='unowned'):
        run.retain_two(root, 'e'*64)
    assert (root/'step-00000001/model').exists()
