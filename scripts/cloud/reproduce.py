"""Focused fresh first-10B launcher; no claim of unattended historical 50B replay."""
import argparse
import subprocess
import sys
from scripts.cloud.fast_start import REPO


def command(args):
    result=[sys.executable,'-m','scripts.cloud.fast_start','--data-dir',args.data_dir,
            '--hardware-profile',args.hardware_profile,'--gpus',str(args.gpus)]
    for key in ['plan','foreground','no_wandb']:
        if getattr(args,key):result.append('--'+key.replace('_','-'))
    if args.run_name:result += ['--run-name',args.run_name]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--hardware-profile',choices=['4090-24','5090-32'],default='4090-24')
    p.add_argument('--gpus',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--through',choices=['10B'],default='10B',help='Only the first stage is automated; see docs/training.md for the actual three-stage history')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--no-wandb',action='store_true')
    p.add_argument('--run-name')
    a=p.parse_args()
    if a.hardware_profile=='5090-32' and a.gpus!=4:
        p.error('The documented 5090 recipe uses --gpus 4; use run_vast directly for other qualified layouts')
    subprocess.run(command(a),cwd=REPO,check=True)


if __name__=='__main__':main()
