"""Finite gradient-component timing; no optimizer update or production state."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from microllama.training import mac_mlx as graph
from microllama.training import mac_ultra_run as ultra
from microllama.training import mac_run as base
from microllama.training.mac import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--embedding', choices=['existing','native'], default='existing')
    p.add_argument('--capture', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(64*2**30)
    mx.set_cache_limit(512*2**20)
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    plan, old = ultra.load_plan(args.plan)
    graph.attention = lambda q,k,v,tile: mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-.5,mask='causal')
    if args.embedding == 'native':
        graph.ordered_take = lambda weight,ids: weight[ids]
    with base.device_lock(Path(old['archive_cache']).parent):
        engine, state = ultra.import_vast(plan,1)
        rows = base.stager_for(old,0).reader(state['data']).next_batch(1)
        indices = engine.indices(rows)
        batch = mx.array(rows)
        backbone = mx.compile(lambda p,b,i: graph.hidden_state(p,engine.config,b[:,:-1],i,
                                                             compute='bfloat16',checkpoint=False))
        labels = batch[:,1:].reshape(-1)
        mask = ((batch[:,:-1]!=7)&(batch[:,:-1]!=0)&(batch[:,1:]!=0)).reshape(-1)
        def head(w,h):
            flat = h.reshape(-1,h.shape[-1])
            result = mx.array(0.,mx.float32)
            for first in range(0,len(labels),1024):
                logits = (flat[first:first+1024].astype(mx.bfloat16)@w.astype(mx.bfloat16).T).astype(mx.float32)
                result = result+mx.sum(mx.fast.cross_entropy(logits,labels[first:first+1024])*mask[first:first+1024])
            return result
        head_grad = mx.compile(mx.value_and_grad(head,argnums=(0,1)))
        backbone_grad = mx.compile(mx.grad(lambda p,b,i,g: mx.sum(backbone(p,b,i)*g)))
        measurements=[]
        for trial in range(3):
            if args.capture and trial==2:
                mx.metal.start_capture(str(root/'gradient.gputrace'))
            started=time.monotonic();h=backbone(engine.params,batch,indices);mx.eval(h)
            forward=time.monotonic()-started
            started=time.monotonic();value,(dw,dh)=head_grad(engine.params['model.embed_tokens.weight'],h);mx.eval(value,dw,dh)
            head_seconds=time.monotonic()-started
            started=time.monotonic();gradients=backbone_grad(engine.params,batch,indices,dh);mx.eval(gradients)
            body_seconds=time.monotonic()-started
            item={'trial':trial,'backbone_forward_seconds':forward,'vocabulary_head_forward_backward_seconds':head_seconds,
                  'backbone_forward_backward_seconds':body_seconds,'loss':float(value)/int(np.asarray(mask).sum())}
            measurements.append(item);print(json.dumps(item),flush=True)
            del gradients,dw,dh,h,value
            if args.capture and trial==2:
                mx.metal.stop_capture()
        write_json(root/'result.json',{'kind':'component-timing-diagnostic','official_targets_advanced':0,
                                      'measurements':measurements,'production_started':False})


if __name__=='__main__':main()
