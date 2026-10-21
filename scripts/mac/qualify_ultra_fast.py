"""Finite original-Vast qualification of the fast MLX graph. Never trains onward."""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import torch

from microllama.training import mac_fast_checkpoint as checkpoint
from microllama.training import mac_ultra_run as ultra, mac_mlx_run as native, mac_run as base
from microllama.training.mac_mlx_fast import FastEngine
from microllama.training.mac import FINAL_MANIFEST,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blocks',type=int,choices=[8,128],default=128)
    args=p.parse_args()
    torch.set_num_threads(8);mx.set_default_device(mx.gpu)
    mx.set_memory_limit(64*2**30);mx.set_cache_limit(8192*2**20)
    plan,old=ultra.load_plan(args.plan)
    root=base.external(args.output);root.mkdir(exist_ok=False)
    def report(item):
        item={'unix':time.time(),**item};write_json(root/'status.json',item)
        print(json.dumps(item),flush=True)
    def release():gc.collect();mx.clear_cache()
    with base.device_lock(Path(old['archive_cache']).parent):
        original,state=ultra.import_vast(plan,1)
        engine=FastEngine(original.params,original.slots,original.policy,original.config,
                          gpu_embedding=False,extras=original.extras)
        del original
        contract={'kind':'finite-fast-qualification','initial_parent_kind':'original-vast-cuda',
                  'parent_manifest_sha256':FINAL_MANIFEST,'data_plan_sha256':sha256(args.plan),
                  'hardware':native.hardware(),'fast_backend':checkpoint.identity(engine),
                  'blocks_per_source':args.blocks,'global_batch':16,'sequence_length':8192,
                  'source_sha256':sha256(Path(__file__)),'production_started':False,
                  'cache_limit_bytes':8192*2**20,'memory_limit_bytes':64*2**30}
        write_json(root/'qualification-plan.json',contract)
        plan_sha=sha256(root/'qualification-plan.json')
        state=copy.deepcopy(state)
        state['config'].update(micro_batch_size=1,gradient_accumulation_steps=16,
                               attention_query_tile=512,loss_chunk_size=1024)
        stager=base.stager_for(old,0);reader=stager.reader(state['data'])
        oracle=json.loads((args.oracle/'result.json').read_text())
        report({'phase':'full8k-independent-derivative-check'})
        ultra.derivative_check(engine,reader,args.oracle,oracle,root/'derivative-check.json')
        spec=copy.deepcopy(plan['validation'])
        if args.blocks!=128:spec.update(blocks=args.blocks,minimum_valid_targets=1)
        validator=native.Validation(spec,stager);validator.progress=report
        report({'phase':'baseline-heldout'})
        baseline=validator(engine);write_json(root/'baseline.json',baseline)
        if args.blocks==128:base.check_losses(baseline,plan['validation']['final_reference'],.05,absolute=True)
        metrics=[]
        for trial in range(2):
            metrics.append(ultra.advance(engine,reader,state,lambda item:report({'phase':'disposable-update','trial':trial,**item})))
            if trial==0:
                fixture=checkpoint.publish(root/'restart-fixture',engine,state,plan_sha,plan['parent'])
        expected=native.signatures(engine,state)
        del engine;release()
        engine,restored=checkpoint.restore(fixture,plan_sha);reader.load_state_dict(restored['data'])
        replay=ultra.advance(engine,reader,restored,lambda item:report({'phase':'gpu-restart-replay',**item}))
        actual=native.signatures(engine,restored)
        comparison={k+'_exact':expected[k]==actual[k] for k in expected}
        comparison['rows_exact']=replay['rows_sha256']==metrics[1]['rows_sha256']
        write_json(root/'restart-comparison.json',{'passed':all(comparison.values()),**comparison})
        if not all(comparison.values()):raise RuntimeError('Fast backend restart was not exact')
        report({'phase':'after-heldout'})
        after=validator(engine);write_json(root/'after.json',after)
        base.check_losses(after,{s:v['loss'] for s,v in baseline.items()},plan['max_validation_increase'])
        del engine;release()
        if checkpoint.bindings()!=contract['fast_backend']['sources']:
            raise RuntimeError('Sources changed during qualification')
        result={'kind':'finite-fast-qualification','passed':True,'full_128_block_validation':args.blocks==128,
                'production_started':False,'official_targets_advanced':0,'plan_sha256':plan_sha,
                'parent_manifest_sha256':FINAL_MANIFEST,'metrics':metrics,'baseline':baseline,'after':after,
                'restart':comparison,'embedding_reduction':'deterministic-cpu',
                'download_bytes_requested':stager.download_bytes,
                'does_not_authorize_production':True}
        write_json(root/'qualification.json',result)
        report({'phase':'completed','passed':True,'production_started':False})


if __name__=='__main__':main()
