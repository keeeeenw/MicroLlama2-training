"""Fast backend identity and exact model/Adam/data/RNG replay after publication."""
import copy
import json
from types import SimpleNamespace

import pytest

mx=pytest.importorskip('mlx.core',exc_type=ImportError)
from microllama.training import mac_fast_checkpoint as checkpoint, mac_mlx_run as native
from microllama.training.mac_mlx import adam_update
from microllama.training.mac_mlx_fast import FastEngine
from test_mac_ultra_run import fixture


@pytest.mark.parametrize('device',['cpu','gpu'])
@pytest.mark.parametrize('kind',['qualification-only','mac-production'])
def test_fast_backend_restores_exact_next_update_and_options(tmp_path,monkeypatch,device,kind):
    previous=mx.default_device();mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
    monkeypatch.setattr(native.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    try:
        initial,rows,reference=fixture(4)
        engine=FastEngine(initial.params,initial.slots,initial.policy,initial.config,
                          micro_batch_size=4,chunk=4,gpu_embedding=True,accumulation_window=2)
        assets=tmp_path/'assets';assets.mkdir();reference.save_pretrained(assets,safe_serialization=True)
        for name in native.ASSETS:
            if not (assets/name).exists():(assets/name).write_text('{}')
        state={'step':1,'tokens':10,'input_positions':8,'data':{'cursor':12},
               'initial_parent_kind':'original-vast-cuda','validation':{},
               'config':{'micro_batch_size':4,'gradient_accumulation_steps':4,
                         'attention_query_tile':512,'loss_chunk_size':4}}
        def step(engine,state):
            _,g,count=engine.gradients(rows)
            engine.params,engine.slots,norm=adam_update(engine.params,g,engine.slots,engine.policy)
            mx.eval(engine.params,engine.slots,norm)
            state.update(step=state['step']+1,tokens=state['tokens']+count,
                         data={'cursor':state['data']['cursor']+16})
        step(engine,state)
        path=checkpoint.publish(tmp_path/'checkpoints',engine,state,'f'*64,assets,kind=kind)
        wrong='mac-production' if kind=='qualification-only' else 'qualification-only'
        with pytest.raises(ValueError):checkpoint.restore(path,'f'*64,kind=wrong)
        step(engine,state); expected=native.signatures(engine,state)
        restored,saved=checkpoint.restore(path,'f'*64,kind=kind)
        assert checkpoint.identity(restored)==checkpoint.identity(engine)
        step(restored,saved)
        assert native.signatures(restored,saved)==expected
        # Changing backend metadata is refused even before payload restoration.
        meta=json.loads((path/'state.json').read_text())
        meta['fast_backend']['backend']='old-graph'
        (path/'state.json').write_text(json.dumps(meta))
        with pytest.raises(ValueError,match='identity'):
            checkpoint.restore(path,'f'*64)
    finally:mx.set_default_device(previous)


def test_legacy_checkpoint_is_not_silently_treated_as_fast(tmp_path):
    (tmp_path/'state.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):
        checkpoint.restore(tmp_path,'f'*64)
