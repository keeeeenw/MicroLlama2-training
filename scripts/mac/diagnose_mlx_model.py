"""BF16 CPU/MPS/MLX gradient triangulation on saved weights; no updates."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import types
from contextlib import nullcontext

import numpy as np
import torch

from microllama.training.mac import write_json,sha256
from microllama.training.mac_run import device_lock, external, stager_for
from microllama.training.mac_attention import install_tiled_attention
from microllama.training.engram import load_engram_model
from microllama.training.mac_gradient_check import full_gradient_check
from microllama.training.objective import chunked_loss, target_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--length',type=int,default=1024)
    p.add_argument('--cpu-only',action='store_true');p.add_argument('--tile',type=int,default=256)
    p.add_argument('--sdpa-math',action='store_true')
    p.add_argument('--precision',choices=['bfloat16','float32'],default='bfloat16')
    args=p.parse_args();root=external(args.output);root.mkdir(exist_ok=False)
    if not 16 <= args.length <= 8192:raise ValueError('Invalid diagnostic length')
    torch.set_num_threads(4)
    with (nullcontext() if args.cpu_only else device_lock(root.parent)):
        plan=json.loads(args.plan.read_text())
        saved=torch.load(args.checkpoint/'mac_state.pt',map_location='cpu',weights_only=True,mmap=True)
        rows=stager_for(plan,0).reader(saved['data']).next_batch(1)[:,:args.length+1]
        del saved
        results={'kind':'diagnostic-only','length':args.length,'official_targets_advanced':0,
                 'rows_sha256':hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest(),'backends':{},
                 'parent_manifest_sha256':sha256(args.checkpoint/'mac_manifest.json'),
                 'parent_plan_sha256':sha256(args.plan),'code_sha256':sha256(Path(__file__)),
                 'precision':args.precision,'query_tile':args.tile,'loss_chunk':256}
        results['attention_backend']='math' if args.sdpa_math else 'default-fused'
        model=load_engram_model(args.checkpoint).float().train()
        model.config.use_cache=False;install_tiled_attention(model,args.tile)
        names=sorted(name for name,_ in model.named_parameters());config=model.config.to_dict()
        for device in (['cpu'] if args.cpu_only else ['cpu','mps']):
            write_json(root/'status.json',{'phase':'gradient','backend':device,'unix':time.time()})
            model.zero_grad(set_to_none=True);model.to(device)
            original=model.model.rotary_emb.forward
            if device=='mps':
                def rotary(self,*values,**kwargs):
                    with torch.autocast('mps',enabled=False):return original(*values,**kwargs)
                model.model.rotary_emb.forward=types.MethodType(rotary,model.model.rotary_emb)
            batch=torch.as_tensor(rows,device=device);x,y=batch[:,:-1],batch[:,1:]
            mask=target_mask(x,y,7,0);count=int(mask.sum().cpu());start=time.monotonic()
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with (sdpa_kernel(SDPBackend.MATH) if args.sdpa_math else nullcontext()):
                with torch.autocast(device,dtype=torch.bfloat16,enabled=args.precision=='bfloat16'):
                    value=chunked_loss(model,x,y,chunk_size=256,loss_mask=mask,reduction='sum')
                value.backward()
            for parameter in model.parameters():parameter.grad.div_(count)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            entry={'loss':float(value.detach().cpu())/count,'gradient_norm':float(norm.cpu()),'seconds':time.monotonic()-start}
            entry['gradients']=full_gradient_check({n:p.grad for n,p in model.named_parameters()},
                export=root/'cpu-reference' if device=='cpu' else None,
                reference=root/'cpu-reference' if device=='mps' else None)
            results['backends'][device]=entry;write_json(root/'results.json',results)
            print(json.dumps({'backend':device,**{k:v for k,v in entry.items() if k!='gradients'},
                  'gradient_check':{k:v for k,v in entry['gradients'].items() if k!='per_parameter'}}),flush=True)
            del value,batch,x,y,mask,norm
        del model;gc.collect()
        if args.cpu_only:
            write_json(root/'status.json',{'phase':'completed','cpu_only':True,'unix':time.time()});return
        torch.mps.empty_cache()
        import mlx.core as mx
        from microllama.training.mac_mlx import loss,memory_indices,scaled_value_and_grad
        mx.set_default_device(mx.gpu);mx.set_memory_limit(16*2**30);mx.set_cache_limit(512*2**20)
        loaded=mx.load(str(args.checkpoint/'model.safetensors'))
        params={name:loaded[name].astype(mx.float32) for name in names};del loaded
        sizes=[params[f'model.engram.{order}_tables.{h}.weight'].shape[0]
               for order in ['bigram','trigram'] for h in range(config['engram_hash_heads'])]
        indices=memory_indices(config,rows[:,:-1],sizes)
        fn=scaled_value_and_grad(lambda p,b,i:loss(p,config,b,i,compute=args.precision,query_tile=args.tile,
            chunk=256,checkpoint=False,reduction='sum'),compute=args.precision,compiled=True,loss_scale=1)
        write_json(root/'status.json',{'phase':'gradient','backend':'mlx','unix':time.time()})
        start=time.monotonic();value,grads=fn(params,mx.array(rows),indices)
        grads={n:g/count for n,g in grads.items()};norm=mx.sqrt(sum(mx.sum(g*g) for g in grads.values()))
        grads={n:g*mx.minimum(1.,1./(norm+1e-6)) for n,g in grads.items()};mx.eval(value,grads,norm)
        entry={'loss':float(value)/count,'gradient_norm':float(norm),'seconds':time.monotonic()-start,
               'gradients':full_gradient_check(grads,reference=root/'cpu-reference')}
        results['backends']['mlx']=entry;write_json(root/'results.json',results)
        print(json.dumps({'backend':'mlx',**{k:v for k,v in entry.items() if k!='gradients'},
              'gradient_check':{k:v for k,v in entry['gradients'].items() if k!='per_parameter'}}),flush=True)
        write_json(root/'status.json',{'phase':'completed','unix':time.time()})


if __name__=='__main__':main()
