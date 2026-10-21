"""Finite dense BF16 GEMM and DRAM-stream measurements; no training/model load."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from microllama.training import mac_run as base
from microllama.training.mac import write_json,sha256


def timed(fn,*args):
    # Frequency ramp and fresh-buffer costs distort very short GEMM screens.
    warmup=time.monotonic()
    while time.monotonic()-warmup<1.0:
        out=fn(*args);mx.eval(out)
    records=[]
    for trial in range(20):
        started=time.monotonic();out=fn(*args);mx.eval(out)
        records.append(time.monotonic()-started)
    return records,statistics.median(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=base.external(args.output);root.mkdir(exist_ok=False)
    mx.set_default_device(mx.gpu);mx.set_memory_limit(64*2**30);mx.set_cache_limit(512*2**20)
    mx.random.seed(271)
    result={'kind':'empirical-roofline-not-published-silicon-peak','device':mx.device_info(mx.gpu),
            'source_sha256':sha256(Path(__file__)),'warmup_seconds_per_case':1,'measured_trials':20,
            'compute':'BF16 inputs/output, native MLX FP32 GEMM accumulator',
            'production_started':False,'gemms':[]}
    with base.device_lock(root.parent):
        for name,m,n,k in [('square4096',4096,4096,4096),('square8192',8192,8192,8192),
                           ('q-projection',8192,1024,1024),('mlp-up',8192,2816,1024),
                           ('mlp-dw',2816,1024,8192),('vocab-head',1024,65536,1024)]:
            a=mx.random.uniform(-.1,.1,(m,k)).astype(mx.bfloat16)
            b=mx.random.uniform(-.1,.1,(n,k)).astype(mx.bfloat16)
            mx.eval(a,b)
            times,median=timed(mx.compile(lambda a,b:a@b.T),a,b)
            item={'name':name,'M':m,'N':n,'K':k,'seconds':times,'median_seconds':median,
                  'tflops':2*m*n*k/median/1e12}
            result['gemms'].append(item);write_json(root/'result.json',result);print(json.dumps(item),flush=True)
            del a,b;gc.collect();mx.clear_cache()
        a=mx.ones((256*2**20,),mx.float32);mx.eval(a)
        times,median=timed(mx.compile(lambda a:a+1.),a)
        result['stream']={'kind':'1GiB FP32 read+write','seconds':times,'median_seconds':median,
                          'effective_GB_per_second':2*a.nbytes/median/1e9}
        write_json(root/'result.json',result);print(json.dumps(result['stream']),flush=True)


if __name__=='__main__':main()
