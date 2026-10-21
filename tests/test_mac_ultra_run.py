"""Batch accounting, CUDA-parent refusal and new-run native replay."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training import mac_ultra_run as ultra
from microllama.training import mac_mlx_run as native
from microllama.training.mac_mlx import parameters_from_torch, adam_from_torch, adam_update
from microllama.training.optimizers import build_optimizer
from microllama.training.mac import update
from test_mac_training import model


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(native.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100*2**30))
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16])
def test_physical_batch_preserves_cuda_global_batch(batch):
    result = ultra.batch_config({'sequence_length': 8192}, batch)
    assert result['micro_batch_size'] * result['gradient_accumulation_steps'] == 16
    assert result['sequence_length'] == 8192


@pytest.mark.parametrize('batch', [0, 3, 32, True, 2.0])
def test_invalid_batch_is_refused(batch):
    with pytest.raises(ValueError, match='global16'):
        ultra.batch_config({}, batch)


@pytest.mark.parametrize('manifest', ['mac_manifest.json', 'mlx_manifest.json'])
def test_existing_mac_checkpoints_are_never_initial_parents(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr(ultra.base, 'external', Path)
    (tmp_path / manifest).write_text('{}')
    with pytest.raises(ValueError, match='Mac checkpoints are forbidden'):
        ultra.verify_vast_parent(tmp_path)


def fixture(batch):
    torch.manual_seed(1901)
    reference = model()
    optimizer = build_optimizer(reference, fused=False, memory_lr_multiplier=5)
    rows = np.tile(np.array([[2, 3, 7, 8, 9, 0, 0, 0], [2, 3, 4, 5, 6, 8, 9, 10]], dtype=np.int32), (8, 1))
    initial = torch.tensor(rows[:1])
    update(reference, optimizer, [(initial[:, :-1], initial[:, 1:])], chunk_size=4)
    slots, policy = adam_from_torch(reference, optimizer, copy.deepcopy(optimizer.state_dict()))
    engine = ultra.BatchEngine(parameters_from_torch(reference), slots, policy, reference.config.to_dict(),
                               query_tile=5, chunk=4, micro_batch_size=batch)
    return engine, rows, reference


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_larger_batch_normalizes_once_by_actual_valid_targets(device):
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    small, rows, _ = fixture(1)
    large, _, _ = fixture(4)
    small_loss, small_gradients, small_count = small.gradients(rows)
    large_loss, large_gradients, large_count = large.gradients(rows)
    expected_count = int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
    assert small_count == large_count == expected_count
    assert small_loss == pytest.approx(large_loss, abs=.002)
    a = np.concatenate([np.array(small_gradients[n]).reshape(-1) for n in sorted(small_gradients)])
    b = np.concatenate([np.array(large_gradients[n]).reshape(-1) for n in sorted(large_gradients)])
    assert np.linalg.norm(a-b) / np.linalg.norm(a) < .02
    assert np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)) > .999


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_batched_checkpoint_restores_batch_policy_and_exact_next_update(tmp_path, device):
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    engine, rows, reference = fixture(4)
    assets = tmp_path / 'assets'
    assets.mkdir()
    reference.save_pretrained(assets, safe_serialization=True)
    for name in native.ASSETS:
        if not (assets/name).exists():
            (assets/name).write_text('{}')
    state = {'step': 1, 'tokens': 10, 'input_positions': 8, 'data': {'cursor': 12},
             'initial_parent_kind': 'original-vast-cuda', 'validation': {},
             'config': {'micro_batch_size': 4, 'gradient_accumulation_steps': 4,
                        'attention_query_tile': 5, 'loss_chunk_size': 4}}
    def step(engine, state):
        value, gradients, count = engine.gradients(rows)
        engine.params, engine.slots, norm = adam_update(engine.params, gradients, engine.slots, engine.policy)
        mx.eval(engine.params, engine.slots, norm)
        state.update(step=state['step']+1, tokens=state['tokens']+count,
                     data={'cursor': state['data']['cursor']+16})
    step(engine, state)
    saved = native.publish(tmp_path/'checkpoints', engine, state, 'f'*64, assets, kind='qualification-only')
    step(engine, state)
    expected = native.signatures(engine, state)
    restored, saved_state = ultra.restore_new_checkpoint(saved, 'f'*64)
    assert restored.micro_batch_size == 4
    step(restored, saved_state)
    assert native.signatures(restored, saved_state) == expected
