"""Isolated full-context GQA derivative checks; no checkpoint or training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from microllama.training.mac import write_json
from microllama.training.mac_run import device_lock, external


def metrics(actual, expected):
    a, b = np.asarray(actual, np.float64).reshape(-1), np.asarray(expected, np.float64).reshape(-1)
    return {'cosine': float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
            'relative_l2': float(np.linalg.norm(a-b) / np.linalg.norm(b)),
            'maximum_absolute_error': float(np.max(np.abs(a-b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--implementation', choices=['native', 'cuda-opmath'], default='cuda-opmath')
    args = parser.parse_args()
    root = external(args.output); root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    with device_lock(root.parent):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(12 * 2**30)
        result = []
        for length, tile in [(128, 32), (1024, 256), (8192, 512)]:
            rng = np.random.default_rng(178 + length)
            source = [rng.normal(size=(1, heads, length, 64)).astype(np.float32) for heads in [4, 2, 2]]
            upstream = rng.normal(size=source[0].shape).astype(np.float32) * .01
            for dtype, mdtype in [(torch.float32, mx.float32), (torch.bfloat16, mx.bfloat16)]:
                write_json(root / 'status.json', {'phase': 'checking', 'length': length, 'dtype': str(dtype)})
                primals = [torch.tensor(x, requires_grad=True) for x in source]
                q, k, v = [x.to(dtype) for x in primals]
                k, v = [x.repeat_interleave(2, dim=1) for x in [k, v]]
                pieces = []
                for first in range(0, length, tile):
                    last = min(first + tile, length)
                    mask = torch.arange(last)[None, :] <= torch.arange(first, last)[:, None]
                    pieces.append(checkpoint(lambda a,b,c,m: F.scaled_dot_product_attention(a,b,c,attn_mask=m),
                        q[...,first:last,:], k[...,:last,:], v[...,:last,:], mask, use_reentrant=False))
                reference = torch.cat(pieces, dim=-2)
                (reference.float() * torch.tensor(upstream)).sum().backward()
                expected = [p.grad.numpy() for p in primals]
                for fused in ([False, True] if args.implementation == 'native' else [True]):
                    def function(q, k, v):
                        q, k, v = [x.astype(mdtype) for x in [q,k,v]]
                        if args.implementation == 'cuda-opmath':
                            from microllama.training.mac_mlx import attention
                            output = attention(q,k,v,tile)
                            return mx.sum(output.astype(mx.float32)*mx.array(upstream)),output.astype(mx.float32)
                        pieces = []
                        for first in range(0,length,tile):
                            last = min(first+tile,length)
                            def part(a,b,c):
                                return mx.fast.scaled_dot_product_attention(a,b,c,scale=.125,mask='causal',force_fused=fused)
                            pieces.append(mx.checkpoint(part)(q[...,first:last,:],k[...,:last,:],v[...,:last,:]))
                        output = mx.concatenate(pieces,axis=-2)
                        return mx.sum(output.astype(mx.float32) * mx.array(upstream)), output.astype(mx.float32)
                    fn = mx.value_and_grad(function, argnums=(0,1,2))
                    (value, output), grads = fn(*[mx.array(x) for x in source])
                    mx.eval(value,output,grads)
                    entry = {'length':length,'tile':tile,'dtype':str(dtype),'force_fused':fused,
                             'implementation':args.implementation,
                             'forward':metrics(np.array(output),reference.detach().float().numpy()),
                             'gradients':{name:metrics(np.array(a),b) for name,a,b in zip(['q','k','v'],grads,expected)}}
                    result.append(entry); write_json(root/'results.json',result)
                    print(json.dumps(entry),flush=True)
                del primals, q,k,v,reference,expected
        write_json(root/'status.json',{'phase':'completed','checks':len(result),'unix':time.time()})


if __name__ == '__main__':
    main()
