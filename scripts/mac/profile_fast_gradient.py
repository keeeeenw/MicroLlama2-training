"""Four finite gradients of the selected MLX backend; no optimizer updates."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import torch

from microllama.training import mac_ultra_run as ultra,mac_run as base
from microllama.training.mac_mlx_fast import FastEngine
from microllama.training.mac import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    mx.set_default_device(mx.gpu);mx.set_memory_limit(64*2**30);mx.set_cache_limit(512*2**20)
    torch.set_num_threads(8)
    plan,old=ultra.load_plan(args.plan)
    root=base.external(args.output);root.mkdir(exist_ok=False)
    with base.device_lock(Path(old['archive_cache']).parent):
        original,state=ultra.import_vast(plan,1)
        engine=FastEngine(original.params,original.slots,original.policy,original.config,
                          extras=original.extras,gpu_embedding=True)
        rows=base.stager_for(old,0).reader(state['data']).next_batch(1)
        batch=mx.array(rows);indices=engine.indices(rows);mx.eval(batch,indices)
        measurements=[]
        for trial in range(4):
            started=time.monotonic();value,g=engine.gradient(engine.params,batch,indices);mx.eval(value,g)
            item={'trial':trial,'seconds':time.monotonic()-started,'summed_nll':float(value)}
            measurements.append(item);print(json.dumps(item),flush=True)
            del g,value
        write_json(root/'result.json',{'kind':'finite-gradient-profile','production_started':False,
                                      'measurements':measurements,'optimizer_updates':0})


if __name__=='__main__':main()
