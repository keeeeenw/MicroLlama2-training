"""Check native likelihoods against direct, single-example, and CPU/CUDA scoring."""
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path

from microllama.data_io.download import sha256
from microllama.data_io.transfer import atomic_json
from scripts.evaluation.checkpoint_identity import manifest_path, load_model

TOLERANCE = .001
PAIRS = [('A triangle has',' three sides'),('The capital of France is',' Paris'),
         ('The water cycle describes the movement of',' water'),('One plus one equals',' two')]


def scores(model, tokenizer, device):
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    lm=HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',device=device,batch_size=4,
            dtype='float32',max_length=model.config.max_position_embeddings,
            add_bos_token=False,prefix_token_id=model.config.bos_token_id)
    requests=[Instance(request_type='loglikelihood',doc={},arguments=x,idx=i) for i,x in enumerate(PAIRS)]
    batch=lm.loglikelihood(requests);lm.batch_size_per_gpu=1;single=lm.loglikelihood(requests)
    rows=[]
    with torch.inference_mode():
        for pair,b,s in zip(PAIRS,batch,single):
            ctx,cont=lm._encode_pair(*pair);ids=ctx+cont
            logits=model(torch.tensor([ids[:-1]],device=device),use_cache=False).logits[0,len(ctx)-1:]
            direct=torch.log_softmax(logits.float(),dim=-1).gather(-1,torch.tensor(cont,device=device)[:,None]).sum().item()
            values=[b[0],s[0],direct]
            if not all(math.isfinite(x) for x in values) or max(values)-min(values)>TOLERANCE or b[1]!=s[1]:
                raise ValueError(f'Native likelihood parity failed on {device}: {values}')
            rows.append({'context':pair[0],'continuation':pair[1],'batch4':b[0],
                         'batch1':s[0],'direct':direct,'greedy':b[1]})
    return rows


def run(args):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.device=='cpu':assert not torch.cuda.is_available()
    else:
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    model=load_model(args.checkpoint).float().eval()
    graft=None
    try:
        if args.graft_reader:
            from microllama.evaluation.graft import load_graft_reader
            graft=load_graft_reader(model,args.checkpoint,args.graft_reader,args.graft_bank)
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
        rows=scores(model,tokenizer,'cpu')
        if args.device=='cuda':
            cpu=rows
            model.to('cuda')
            if graft:
                graft.injection.reader.to('cuda')
                graft.injection.bank.to('cuda')
                graft.metadata['inference_device']='cuda'
            rows=scores(model,tokenizer,'cuda')
            for row,reference in zip(rows,cpu):
                error=abs(row['batch4']-reference['batch4'])
                if error>TOLERANCE or row['greedy']!=reference['greedy']:
                    raise ValueError(f'CPU/CUDA likelihood disagreement: {error}')
                row.update(cpu_batch4=reference['batch4'],cpu_batch1=reference['batch1'],
                           cpu_direct=reference['direct'],cpu_cuda_absolute_error=error)
        atomic_json(args.output,{'passed':True,'device':args.device,'dtype':str(next(model.parameters()).dtype),
            'architecture':model.config.model_type,'max_length':model.config.max_position_embeddings,
            'add_bos_token':False,'prefix_token_id':model.config.bos_token_id,
            'tolerance':TOLERANCE,'tf32':False,'cpu_cuda_checked':args.device=='cuda','scores':rows,
            'manifest_sha256':sha256(manifest_path(args.checkpoint)),
            **({'graft':graft.metadata} if graft else {})})
        print(f'Native likelihood parity passed on {args.device}',flush=True)
    finally:
        if graft:graft.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--graft-reader',type=Path)
    p.add_argument('--graft-bank',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--gpu-lock-fd',type=int,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.threads<1:p.error('Positive threads required')
    if bool(args.graft_reader)!=bool(args.graft_bank):p.error('Grafting requires both reader and bank')
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if args.device=='cuda':
        from microllama.training.gpu import gpu_lock,idle_gpu
        guard=gpu_lock(args.gpu_lock_fd)
    else:guard=nullcontext()
    with guard:
        if args.device=='cuda' and args.gpu_lock_fd is None:idle_gpu()
        run(args)


if __name__=='__main__':main()
