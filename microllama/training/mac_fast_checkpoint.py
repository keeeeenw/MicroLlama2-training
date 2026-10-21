"""Explicit source/option identity for qualified fast-backend checkpoints.

A fast checkpoint cannot silently restore the old graph.
"""
import copy
import json
from pathlib import Path

from . import mac_mlx_run as native, mac_ultra_run as ultra
from .mac import sha256
from .mac_mlx_fast import FastEngine

BACKEND='mlx-native-fast-v1'


def bindings():
    root=Path(__file__).resolve().parents[1]
    return {**ultra.bindings(), **{n:sha256(root/n) for n in (
        'training/mac_mlx_fast.py','training/mac_gpu_embedding.py',
        'training/metal/embedding_scatter.metal','training/mac_fast_checkpoint.py')}}


def identity(engine):
    return {'backend':BACKEND,'sources':bindings(),'options':{
        'micro_batch_size':engine.micro_batch_size,'chunk':engine.chunk,
        'checkpoint':engine.checkpoint,'accumulation_window':engine.accumulation_window,
        'fused_projections':engine.fused_projections,'gpu_embedding':engine.gpu_embedding}}


def publish(root,engine,state,plan_sha,assets_from,*,kind='qualification-only'):
    if kind not in ('qualification-only','mac-production'):
        raise ValueError('Explicit qualification/production checkpoint kind required')
    if state.get('initial_parent_kind')!='original-vast-cuda':
        raise ValueError('Fast qualification must descend from original Vast CUDA parent')
    saved=copy.deepcopy(state)
    saved['fast_backend']=identity(engine)
    return native.publish(root,engine,saved,plan_sha,assets_from,kind=kind)


def restore(path,plan_sha,*,kind='qualification-only'):
    if kind not in ('qualification-only','mac-production'):
        raise ValueError('Explicit qualification/production checkpoint kind required')
    saved=json.loads((Path(path)/'state.json').read_text()).get('fast_backend')
    if not saved or saved.get('backend')!=BACKEND or saved.get('sources')!=bindings():
        raise ValueError('Fast backend/source identity differs or is missing')
    options=saved['options']
    if (set(options)!={'micro_batch_size','chunk','checkpoint','accumulation_window','fused_projections','gpu_embedding'}
            or type(options['chunk']) is not int or options['chunk']<1
            or any(type(options[n]) is not bool for n in ('checkpoint','fused_projections','gpu_embedding'))):
        raise ValueError('Invalid fast-backend options')
    old,state=native.restore(path,plan_sha,kind=kind)
    if state.get('initial_parent_kind')!='original-vast-cuda' or state['fast_backend']!=saved:
        raise ValueError('Fast checkpoint provenance differs')
    engine=FastEngine(old.params,old.slots,old.policy,old.config,extras=old.extras,**options)
    if identity(engine)!=saved:
        raise ValueError('Restored fast backend differs')
    return engine,state
