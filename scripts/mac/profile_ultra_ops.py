"""Bounded operator timing with imported weights/full8K shapes; no updates."""
import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import torch

from microllama.training import mac_ultra_run as ultra, mac_run as base
from microllama.training.mac_mlx import embedding, qwen_rms, silu
from microllama.training.mac import write_json, FINAL_MANIFEST


def measure(fn, inputs, trials=6):
    forward = mx.compile(fn)
    _, sample = mx.vjp(fn, inputs, [mx.ones_like(fn(*inputs))])
    mx.eval(sample)
    output = fn(*inputs)
    cotangent = mx.ones_like(output)*.001
    backward = mx.compile(lambda *args: mx.vjp(fn, list(args[:-1]), [args[-1]]))
    mx.eval(inputs, cotangent)
    measurements = []
    for trial in range(trials):
        started = time.monotonic(); result = forward(*inputs); mx.eval(result)
        f = time.monotonic()-started
        started = time.monotonic(); result = backward(*inputs, cotangent); mx.eval(result)
        fb = time.monotonic()-started
        measurements.append({'trial': trial, 'forward_seconds': f, 'forward_backward_seconds': fb})
    return {'trials': measurements,
            'median_forward_seconds': statistics.median(m['forward_seconds'] for m in measurements[1:]),
            'median_forward_backward_seconds': statistics.median(m['forward_backward_seconds'] for m in measurements[1:])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    mx.set_default_device(mx.gpu); mx.set_memory_limit(64*2**30); mx.set_cache_limit(512*2**20)
    torch.set_num_threads(8)
    plan, old = ultra.load_plan(args.plan)
    root = base.external(args.output); root.mkdir(exist_ok=False)
    with base.device_lock(Path(old['archive_cache']).parent):
        engine, state = ultra.import_vast(plan, 1)
        rows = base.stager_for(old, 0).reader(state['data']).next_batch(1)
        params, c = engine.params, engine.config
        x = embedding(params['model.embed_tokens.weight'], mx.array(rows[:, :-1]), c['pad_token_id'])
        weights = {n.removeprefix('model.layers.0.'): v for n, v in params.items() if n.startswith('model.layers.0.')}
        def linear(x, w): return x.astype(mx.bfloat16)@w.astype(mx.bfloat16).T
        def qkv(w, x):
            h = qwen_rms(x, w['input_layernorm.weight'], c['rms_norm_eps'])
            shape = (*x.shape[:2], -1, c['head_dim'])
            q, k, v = [linear(h, w[f'self_attn.{n}_proj.weight']).reshape(shape) for n in ('q','k','v')]
            q = qwen_rms(q, w['self_attn.q_norm.weight'], c['rms_norm_eps']).transpose(0,2,1,3)
            k = qwen_rms(k, w['self_attn.k_norm.weight'], c['rms_norm_eps']).transpose(0,2,1,3)
            q = mx.fast.rope(q, dims=c['head_dim'], traditional=False, base=c['rope_theta'], scale=1., offset=0)
            k = mx.fast.rope(k, dims=c['head_dim'], traditional=False, base=c['rope_theta'], scale=1., offset=0)
            return q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.transpose(0,2,1,3)
        def mlp(w, x):
            h = qwen_rms(x, w['post_attention_layernorm.weight'], c['rms_norm_eps'])
            return x+linear(silu(linear(h,w['mlp.gate_proj.weight']))*linear(h,w['mlp.up_proj.weight']),w['mlp.down_proj.weight']).astype(mx.float32)
        q,k,v = qkv(weights,x); mx.eval(q,k,v,x)
        attn = lambda q,k,v: mx.fast.scaled_dot_product_attention(q,k,v,scale=c['head_dim']**-.5,mask='causal')
        results = {'kind':'operator-timing', 'sequence_length':8192, 'batch':1,
                   'parent_manifest_sha256':FINAL_MANIFEST, 'production_started':False}
        names = [n for n in weights if n.startswith('mlp.') or n == 'post_attention_layernorm.weight']
        mlp_flat = lambda *args: mlp(dict(zip(names,args[:-1])),args[-1])
        for name,fn,inputs in [('attention',attn,[q,k,v]),('mlp',mlp_flat,[*[weights[n] for n in names],x])]:
            results[name] = measure(fn,inputs)
            write_json(root/'result.json',results)
            print(json.dumps({'component':name, **results[name]}),flush=True)


if __name__=='__main__': main()
