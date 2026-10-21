"""Native restart/RNG/publication refusal checks on tiny CPU-only state."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training.mac_mlx import parameters_from_torch, adam_from_torch, adam_update
from microllama.training.mac_mlx_run import Engine, ASSETS, publish, restore, signatures, restore_rng_key
from microllama.training.mac import update
from microllama.training.optimizers import build_optimizer
from test_mac_training import model


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    monkeypatch.setattr('microllama.training.mac_mlx_run.shutil.disk_usage',
                        lambda path: SimpleNamespace(free=100*2**30))
    yield
    mx.set_default_device(previous)


def test_rng_restore_reproduces_next_draw_without_shadowing_sentinel():
    mx.random.seed(813)
    mx.eval(mx.random.uniform(shape=(3,)))
    key = np.array(mx.random.state[0])
    expected = np.array(mx.random.uniform(shape=(16,)))
    restore_rng_key(mx.array(key))
    np.testing.assert_array_equal(np.array(mx.random.uniform(shape=(16,))), expected)
    assert type(mx.random.state).__name__ == '_RandomState'
    with pytest.raises(ValueError, match='layout'):
        restore_rng_key(mx.array([1],dtype=mx.uint32))


def fixture(tmp_path):
    torch.manual_seed(913)
    reference = model()
    optimizer = build_optimizer(reference,fused=False,memory_lr_multiplier=5)
    rows = torch.tensor([[2,3,4,7,8,9,10,11]])
    update(reference,optimizer,[(rows[:,:-1],rows[:,1:])],chunk_size=4)
    slots,policy = adam_from_torch(reference,optimizer,copy.deepcopy(optimizer.state_dict()))
    engine = Engine(parameters_from_torch(reference),slots,policy,reference.config.to_dict(),query_tile=5,chunk=4)
    assets = tmp_path/'assets';assets.mkdir();reference.save_pretrained(assets,safe_serialization=True)
    for name in ASSETS:
        if not (assets/name).exists():(assets/name).write_text('{}')
    state = {'step':1,'tokens':10,'input_positions':8,'data':{'cursor':12},
             'config':{'attention_query_tile':5,'loss_chunk_size':4},'validation':{},'phase':'training'}
    return engine,state,assets


def test_native_checkpoint_preserves_all_state_and_refuses_overwrite_and_tamper(tmp_path):
    engine,state,assets = fixture(tmp_path)
    expected = signatures(engine,state)
    path = publish(tmp_path/'checkpoints',engine,state,'c'*64,assets)
    loaded,restored = restore(path,'c'*64)
    assert signatures(loaded,restored) == expected
    with pytest.raises(ValueError,match='overwrite'):
        publish(tmp_path/'checkpoints',engine,state,'c'*64,assets)
    with pytest.raises(ValueError,match='schema/kind/plan'):
        restore(path,'d'*64)
    with pytest.raises(ValueError,match='schema/kind/plan'):
        restore(path,'c'*64,kind='qualification-only')
    (path/'state.json').write_text('{}')
    with pytest.raises(ValueError,match='payload differs'):
        restore(path,'c'*64)


def test_native_checkpoint_refuses_nonfinite_moments_before_publication(tmp_path):
    engine,state,assets = fixture(tmp_path)
    name = next(iter(engine.slots))
    engine.slots[name]['m'] = engine.slots[name]['m'] * mx.array(float('nan'))
    with pytest.raises(RuntimeError,match='Nonfinite'):
        publish(tmp_path/'checkpoints',engine,state,'c'*64,assets)
    assert not (tmp_path/'checkpoints/step-00000001').exists()


def test_native_checkpoint_keeps_disk_reserve(tmp_path,monkeypatch):
    engine,state,assets = fixture(tmp_path)
    monkeypatch.setattr('microllama.training.mac_mlx_run.shutil.disk_usage',lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='reserve'):
        publish(tmp_path/'checkpoints',engine,state,'c'*64,assets)
    assert not (tmp_path/'checkpoints/step-00000001.partial').exists()


def test_native_bf16_next_update_replays_exactly_after_reload(tmp_path):
    engine,state,assets = fixture(tmp_path)
    rows = np.array([[2,3,4,7,8,9,10,11]],dtype=np.int32)
    def step(engine,state):
        value,gradients = engine.gradient(engine.params,mx.array(rows),engine.indices(rows))
        count = int(((rows[:,:-1]!=7)&(rows[:,:-1]!=0)&(rows[:,1:]!=0)).sum())
        gradients = {n:g/count for n,g in gradients.items()}
        params,slots,_ = adam_update(engine.params,gradients,engine.slots,engine.policy)
        mx.eval(params,slots,value);engine.params,engine.slots = params,slots
        state.update(step=state['step']+1,tokens=state['tokens']+count,
                     data={'cursor':state['data']['cursor']+8})
    step(engine,state)
    path=publish(tmp_path/'checkpoints',engine,state,'e'*64,assets)
    step(engine,state);expected=signatures(engine,state)
    loaded,restored=restore(path,'e'*64)
    step(loaded,restored)
    assert signatures(loaded,restored)==expected
