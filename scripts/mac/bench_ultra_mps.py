"""Three finite full8K MPS microbatch timings from Vast; no saved training."""
import argparse
import json
from pathlib import Path
import time
import types

import torch
import torch.nn.functional as F

from microllama.training import mac_run as base
from microllama.training.mac import load_final, write_json, FINAL_MANIFEST
from microllama.training.objective import chunked_loss, target_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpointing',action='store_true')
    args=p.parse_args();torch.set_num_threads(8)
    root=base.external(args.output);root.mkdir(exist_ok=False)
    old=base.load_plan(args.plan)
    torch.mps.set_per_process_memory_fraction(.9)
    with base.device_lock(Path(old['archive_cache']).parent):
        model,optimizer,state=load_final(old['parent'],torch.device('mps'))
        del state['optimizer']
        model.config.use_cache=False
        model.train()
        if args.checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        for module in model.modules():
            if isinstance(module,torch.nn.Embedding):
                def forward(self,ids):
                    out=F.embedding(ids,self.weight,padding_idx=None)
                    return out if self.padding_idx is None else torch.where(ids.ne(self.padding_idx).unsqueeze(-1),out,out.detach())
                module.forward=types.MethodType(forward,module)
        original=model.model.rotary_emb.forward
        def rotary(self,*values,**kwargs):
            with torch.autocast('mps',enabled=False):return original(*values,**kwargs)
        model.model.rotary_emb.forward=types.MethodType(rotary,model.model.rotary_emb)
        rows=base.stager_for(old,0).reader(state['data']).next_batch(1)
        batch=torch.as_tensor(rows,device='mps');x,y=batch[:,:-1],batch[:,1:]
        mask=target_mask(x,y,7,0);count=int(mask.sum().cpu());measurements=[]
        for trial in range(3):
            optimizer.zero_grad(set_to_none=True);torch.mps.synchronize();started=time.monotonic()
            with torch.autocast('mps',dtype=torch.bfloat16):
                value=chunked_loss(model,x,y,chunk_size=256,loss_mask=mask,reduction='sum')
            value.backward()
            for parameter in model.parameters():parameter.grad.div_(count)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            torch.mps.synchronize();elapsed=time.monotonic()-started
            item={'trial':trial,'full8k_microbatch_seconds':elapsed,'valid_targets_per_second':count/elapsed,
                  'loss':float(value.detach().cpu())/count,'grad_norm':float(norm.cpu()),
                  'driver_bytes':torch.mps.driver_allocated_memory(),'allocated_bytes':torch.mps.current_allocated_memory()}
            measurements.append(item);print(json.dumps(item),flush=True)
        write_json(root/'result.json',{'kind':'microbatch-only-backend-screen','production_started':False,
            'official_targets_advanced':0,'parent_manifest_sha256':FINAL_MANIFEST,'checkpointing':args.checkpointing,
            'no_optimizer_steps':True,'measurements':measurements,'timing_excludes':'Adam and accumulation; screening only'})


if __name__=='__main__':main()
