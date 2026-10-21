"""Launch one frozen BF16 MLX qualify-then-train worker, without retries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from microllama.training.mac import sha256, write_json
from microllama.training.mac_mlx_run import load_plan
from microllama.training.mac_run import external


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--launch-root',type=Path,required=True)
    p.add_argument('--until-stopped',action='store_true',required=True)
    args=p.parse_args();plan,old=load_plan(args.plan)
    root=external(args.launch_root);root.mkdir(exist_ok=False)
    source=Path(__file__).resolve().parents[2];code=root/'code'
    shutil.copytree(source/'microllama',code/'microllama',ignore=shutil.ignore_patterns('__pycache__','*.pyc','._*'))
    for name,expected in plan['implementation'].items():
        target=code/'microllama'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/'microllama'/name,target)
        if sha256(target)!=expected:raise RuntimeError('Frozen MLX source differs from reviewed plan')
    (root/'tmp').mkdir()
    env=dict(os.environ)
    for name in ('WANDB_RUN_ID','WANDB_RESUME','WANDB_NAME','WANDB_RUN_GROUP'):env.pop(name,None)
    env.update(PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HUB_CACHE=str(root/'hf-cache'),HF_XET_CACHE=str(Path(old['archive_cache'])/'xet'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_XET_CHUNK_CACHE_SIZE_BYTES='0',
        HF_HUB_DISABLE_TELEMETRY='1',TMPDIR=str(root/'tmp'),TORCH_HOME=str(root/'torch-cache'),
        XDG_CACHE_HOME=str(root/'cache'),WANDB_SILENT='true')
    cmd=['/usr/bin/caffeinate','-i','-s',sys.executable,'-u','-m','microllama.training.mac_mlx_run',
         'pipeline','--plan',str(args.plan.resolve()),'--output',str(root/'run'),'--until-stopped']
    with (root/'worker.log').open('xb') as stream:
        worker=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'unix':time.time(),'pid':worker.pid,'command':cmd,'plan_sha256':sha256(args.plan),
        'source_snapshot':str(code),'autorestart':False,'until_stopped':True,'max_download_bytes':0,
        'status':str(root/'run/worker-status.json'),'log':str(root/'worker.log'),
        'target_tokens':plan['config']['target_tokens'],'old_cuda_run_reused':False}
    write_json(root/'launch.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
