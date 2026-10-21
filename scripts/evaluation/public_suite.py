"""Float32, zero-shot V1 task suite; CPU default, single-worker CUDA optional."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time
from contextlib import nullcontext

from microllama.data_io.download import sha256
from microllama.data_io.transfer import atomic_json
from scripts.evaluation.checkpoint_identity import manifest_path, verify_checkpoint, load_model as load_checkpoint_model

TASKS = ('hellaswag', 'openbookqa', 'winogrande', 'arc_easy', 'arc_challenge', 'boolq', 'piqa')
METRICS = {task: ('acc,none' if task in ('winogrande', 'boolq') else 'acc_norm,none') for task in TASKS}


def configure_threads(torch, threads):
    """A pool worker may score several tasks; initialize interop only once."""
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)


def evaluate_task(checkpoint, output, task, threads, batch_size, limit, hf_model=False, tokenizer_path=None, add_bos_token=False,
                  graft_reader=None, graft_bank=None, device='cpu'):
    if device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = str(threads)
    import torch
    from transformers import AutoTokenizer
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    configure_threads(torch,threads)
    if device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU evaluation must have no visible GPU'
    else:
        assert torch.cuda.is_available(), 'CUDA evaluation requires an available GPU'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    graft = None
    if hf_model:
        # Use the harness's normal HF path, matching the V1 model-card command.
        # Its model/tokenizer snapshots were separately pinned and verified.
        lm = HFLM(pretrained=checkpoint, tokenizer=tokenizer_path, backend='causal',
                  device=device, batch_size=batch_size, dtype='float32',
                  trust_remote_code=False, add_bos_token=add_bos_token)
    else:
        kind = json.loads((Path(checkpoint)/'config.json').read_text()).get('model_type')
        if kind in ('microllama_hybrid_engram', 'microllama_dense_engram'):
            if device != 'cuda':
                raise ValueError('The hybrid kernel environment currently requires CUDA evaluation')
            from microllama.training.hybrid import load_model, use_reference_delta
            model = load_model(checkpoint).float().to(device).eval()
            if kind == 'microllama_hybrid_engram':
                use_reference_delta(model)
        else:
            model = load_checkpoint_model(checkpoint).float().to(device).eval()
        if graft_reader:
            from microllama.evaluation.graft import load_graft_reader
            graft = load_graft_reader(model, checkpoint, graft_reader, graft_bank)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, backend='causal', device=device,
                  batch_size=batch_size, max_length=model.config.max_position_embeddings,
                  dtype='float32', add_bos_token=False, prefix_token_id=model.config.bos_token_id)
    started = time.time()
    result = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, device=device,
                             batch_size=batch_size, limit=limit, log_samples=True,
                             bootstrap_iters=1000, random_seed=0, numpy_random_seed=1234,
                             torch_random_seed=1234, fewshot_random_seed=1234)
    identity = {'path':str(checkpoint), 'engram':not hf_model, 'device':device, 'dtype':'float32',
                'tf32':False,
                'elapsed_seconds':time.time()-started, 'limit':limit,
                'max_length':lm.max_length, 'prefix_token_id':lm.prefix_token_id,
                'tokenizer_path':str(tokenizer_path or checkpoint),
                'add_bos_token':lm.add_bos_token}
    if not hf_model:
        identity['architecture'] = model.config.model_type
        if model.config.model_type == 'microllama_hybrid_engram':
            identity['delta_kernel'] = 'transformers-fp32-reference'
    if not hf_model:
        identity['manifest_sha256']=sha256(manifest_path(checkpoint))
        identity['checkpoint_format']='mlx' if manifest_path(checkpoint).name=='mlx_manifest.json' else 'cuda'
        if hasattr(model, '_evaluation_metadata_receipt'):
            identity['metadata_receipt']=model._evaluation_metadata_receipt
        if graft:
            identity['graft']=graft.metadata
        result['native_checkpoint']=identity
    result['model_identity']=identity
    # Preserve sample-level outputs and harness task/version identities.
    result = json.loads(json.dumps(result, default=handle_non_serializable))
    atomic_json(Path(output)/f'{task}.json', result)
    metric = METRICS[task]
    metrics = result['results'][task]
    if graft:
        graft.close()
    return {'task':task, 'metric':metric, 'score':metrics[metric],
            'stderr':metrics.get(metric.replace(',none','_stderr,none')),
            'samples':result['n-samples'][task], 'elapsed_seconds':time.time()-started}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path, nargs='?')
    p.add_argument('--hf-model', type=Path, help='Verified local HF snapshot instead of a native checkpoint')
    p.add_argument('--tokenizer', type=Path, help='Verified local tokenizer snapshot; required with --hf-model')
    p.add_argument('--add-bos-token', action='store_true',
                   help='Use the HF tokenizer native special-token processing (HF models only)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--limit', type=int, help='Setup smoke test only; omitted for the full suite')
    p.add_argument('--graft-reader', type=Path, help='Verified reader directory containing reader-state.pt and reader-manifest.json')
    p.add_argument('--graft-bank', type=Path, help='Exact frozen bank paired with --graft-reader')
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--gpu-lock-fd', type=int, help=argparse.SUPPRESS)
    args=p.parse_args()
    if bool(args.checkpoint)==bool(args.hf_model) or bool(args.hf_model)!=bool(args.tokenizer):
        p.error('Supply either a native checkpoint or --hf-model plus --tokenizer')
    if args.add_bos_token and not args.hf_model:
        p.error('--add-bos-token requires --hf-model')
    if bool(args.graft_reader) != bool(args.graft_bank) or (args.graft_reader and args.hf_model):
        p.error('Grafting requires a native checkpoint and both --graft-reader and --graft-bank')
    if (args.output/'status.json').exists():
        p.error('Output already has an evaluation attempt; use a fresh output directory')
    if min(args.workers,args.threads,args.batch_size)<1:
        p.error('Positive workers, threads and batch size required')
    if args.device=='cuda' and args.workers!=1:
        p.error('CUDA suites require --workers 1 to avoid competing GPU processes')
    return args


def run_suite(args):
    if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
    args.output.mkdir(parents=True,exist_ok=True)
    from importlib.metadata import version
    model_path=(args.hf_model or args.checkpoint).resolve()
    if args.hf_model:
        # The download receipt binds upstream commits and all LFS file hashes.
        receipt=json.loads((model_path.parent/'download-receipt.json').read_text())
        expected=receipt['files']
        for relative,metadata in expected.items():
            file=model_path.parent/relative
            if not file.is_file() or file.stat().st_size!=metadata['bytes'] or sha256(file)!=metadata['sha256']:
                raise ValueError(f'Missing or changed verified HF snapshot file: {relative}')
        for directory in (model_path,args.tokenizer.resolve()):
            for file in directory.rglob('*'):
                if file.is_file() and file.name!='.cache' and '.cache' not in file.parts:
                    relative=str(file.relative_to(model_path.parent))
                    if relative not in expected or sha256(file)!=expected[relative]['sha256']:
                        raise ValueError(f'HF snapshot does not match download receipt: {relative}')
        identity={'hf_model':str(model_path),'tokenizer':str(args.tokenizer.resolve()),
                  'download_receipt_sha256':sha256(model_path.parent/'download-receipt.json'),
                  'upstream':receipt['upstream']}
    else:
        digest=verify_checkpoint(model_path)
        manifest=json.loads(manifest_path(model_path).read_text())
        identity={'checkpoint_step':manifest['step'],'checkpoint_tokens':manifest['tokens'],
                  'manifest_sha256':digest,'tokenizer_sha256':sha256(model_path/'tokenizer.json')}
        if args.graft_reader:
            from microllama.evaluation.graft import verify_graft_artifacts
            identity['graft']=verify_graft_artifacts(model_path,args.graft_reader,args.graft_bank)
    protocol={'tasks':list(TASKS),'metrics':METRICS,'num_fewshot':0,'chat_template':False,**identity,
              'harness_version':version('lm_eval'),'datasets_version':version('datasets'),
              'libraries':{name:version(name) for name in ('transformers','huggingface_hub','torch','tokenizers')},
              'dtype':'float32','device':args.device,'tf32':False,'batch_size':args.batch_size,'threads':args.threads,
              'workers':args.workers,'add_bos_token':args.add_bos_token,'prefix_token_id':None if args.hf_model else 1,'limit':args.limit,
              'reference':'https://huggingface.co/keeeeenw/MicroLlama/blob/main/README.md#evaluation'}
    atomic_json(args.output/'protocol.json',protocol)
    state={'phase':'running','started_unix':time.time(),'completed':{},'failed':{},'protocol':protocol}
    def save():
        state['updated_unix']=time.time();atomic_json(args.output/'status.json',state)
        print(json.dumps({k:v for k,v in state.items() if k!='protocol'}),flush=True)
    save()
    with ProcessPoolExecutor(min(args.workers,len(TASKS)),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(evaluate_task,str(model_path),str(args.output),task,args.threads,args.batch_size,args.limit,
                             bool(args.hf_model),str(args.tokenizer.resolve()) if args.tokenizer else None,args.add_bos_token,
                             str(args.graft_reader.resolve()) if args.graft_reader else None,
                             str(args.graft_bank.resolve()) if args.graft_bank else None,args.device):task for task in TASKS}
        for future in as_completed(futures):
            task=futures[future]
            try:state['completed'][task]=future.result()
            except Exception as error:state['failed'][task]=f'{type(error).__name__}: {error}'
            save()
    state['phase']='failed' if state['failed'] else 'completed'
    if not state['failed']:
        state['mean_accuracy']=sum(v['score'] for v in state['completed'].values())/len(TASKS)
    save()
    if state['failed']:raise SystemExit(1)


def main():
    args=parse_args()
    if args.device=='cuda':
        from microllama.training.gpu import gpu_lock, idle_gpu
        guard=gpu_lock(args.gpu_lock_fd)
    else:guard=nullcontext()
    with guard:
        if args.device=='cuda' and args.gpu_lock_fd is None:idle_gpu()
        run_suite(args)


if __name__=='__main__':
    main()
