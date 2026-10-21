"""Finite synthetic full8K native/custom attention timing; no model updates."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from microllama.training import mac_run as base
from microllama.training.mac import write_json, sha256
from microllama.training.mac_nax_attention import attention, kernels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,choices=[256,1024,8192],default=8192)
    p.add_argument('--batch',type=int,choices=[1,2,4],default=1)
    args=p.parse_args()
    mx.set_default_device(mx.gpu); mx.set_memory_limit(64*2**30); mx.set_cache_limit(512*2**20)
    root=base.external(args.output);root.mkdir(exist_ok=False)
    with base.device_lock(root.parent):
        mx.random.seed(831)
        inputs=[mx.random.normal((args.batch,16,args.length,64)).astype(mx.bfloat16) for _ in range(4)]
        mx.eval(inputs)
        native=lambda q,k,v: mx.fast.scaled_dot_product_attention(q,k,v,scale=.125,mask='causal')
        result={'kind':'isolated-attention-benchmark','production_started':False,
                'shape':list(inputs[0].shape),'variants':{},'source_sha256':sha256(Path(__file__)),
                'kernel_sources':{str(p.name):sha256(p) for p in Path('microllama/training/metal').glob('*')
                                  if p.is_file() and not p.name.startswith('.')},
                'operator_sha256':sha256(Path('microllama/training/mac_nax_attention.py'))}
        reference=None
        for name,fn in [('native',native),('custom',attention)]:
            backward=mx.compile(lambda q,k,v,g: mx.vjp(fn,[q,k,v],[g]))
            measurements=[]
            for trial in range(6):
                mx.reset_peak_memory();started=time.monotonic()
                output,gradients=backward(*inputs);mx.eval(output,gradients)
                measurements.append({'trial':trial,'forward_backward_seconds':time.monotonic()-started,
                                     'peak_bytes':mx.get_peak_memory()})
            arrays=[np.asarray(g.astype(mx.float32)) for g in gradients]
            if reference is None: reference=arrays
            else:
                result['gradient_comparison']={}
                for label,a,b in zip(['dq','dk','dv'],arrays,reference):
                    av,bv=a.reshape(-1).astype(np.float64),b.reshape(-1).astype(np.float64)
                    result['gradient_comparison'][label]={'relative_l2':float(np.linalg.norm(av-bv)/np.linalg.norm(bv)),
                        'cosine':float(np.dot(av,bv)/(np.linalg.norm(av)*np.linalg.norm(bv)))}
                _,again=backward(*inputs);mx.eval(again)
                result['repeated_gradient_exact']=all(np.array_equal(a,np.asarray(b.astype(mx.float32))) for a,b in zip(arrays,again))
            result['variants'][name]={'measurements':measurements,
                'median_seconds':statistics.median(m['forward_backward_seconds'] for m in measurements[1:])}
            write_json(root/'result.json',result)
            print(json.dumps({'variant':name,**result['variants'][name]}),flush=True)
        result['speedup']=result['variants']['native']['median_seconds']/result['variants']['custom']['median_seconds']
        write_json(root/'result.json',result)
        print(json.dumps({'speedup':result['speedup'],'comparison':result['gradient_comparison'],
                          'exact':result['repeated_gradient_exact']}),flush=True)


if __name__=='__main__':main()
