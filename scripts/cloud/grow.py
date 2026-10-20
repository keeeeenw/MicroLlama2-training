"""Idempotent cron gate: completed 10B -> 40 updates -> 250M pilot -> 50B.

Runs without a daemon. Each launch detaches through run_vast, and durable
attempt receipts prevent retries after an ambiguous launch or failed gate.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from microllama.data_io.growth_reader import MixtureTokenBlocks
from microllama.data_io.transfer import atomic_json

REPO = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(Path(path).read_text())

def alive(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return bool(pid and path.exists() and b'microllama.training.distributed' in path.read_bytes())

def healthy(status, world):
    if status.get('error') or status.get('wandb_error') or status.get('phase')=='failed':
        raise RuntimeError(f"Training failure: {status.get('error') or status.get('wandb_error')}")
    if status.get('world_size') != world or status.get('architecture') != 'engram':
        raise ValueError('Expected ranks and Engram required')
    for key in ('loss','validation_loss'):
        if not math.isfinite(status.get(key,float('nan'))):
            raise ValueError(f'Nonfinite/missing {key}')
    if not status.get('wandb_url'):
        raise ValueError('Tracked training run required')

def verify(path, settings):
    # Use the pinned training environment; no CUDA allocations are needed.
    result=subprocess.run([settings['train_python'],'-m','scripts.cloud.grow','--inspect-checkpoint',
        str(path)],cwd=REPO,check=True,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    return json.loads(result.stdout)

def checkpoint_report(path):
    import torch
    from microllama.training.distributed import verify_checkpoint
    checksum=verify_checkpoint(path)
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
    for group in state['optimizer']['state'].values():
        for value in group.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite saved optimizer state')
    data=state['data']; result={'manifest_sha256':checksum,'step':state['step'],'tokens':state['tokens'],
        'config':state['config'],'data_mode':data['mode'],'source_targets':data.get('source_targets'),
        'parent_tokens':data.get('parent_tokens'),'world_size':state['distributed']['world_size'],
        'architecture':state['distributed']['architecture']}
    if data.get('source_targets') is not None and data['parent_tokens']+sum(data['source_targets'].values())!=state['tokens']:
        raise ValueError('Checkpoint source/historical counters do not reconcile')
    return result

def bound_checkpoint(status,settings):
    result=verify(Path(status['checkpoint']),settings)
    for key in ('step','tokens','config','world_size','architecture'):
        if result[key]!=status[key]: raise ValueError(f'Checkpoint/status mismatch: {key}')
    return result

def metric_gate(metrics, start_step, end_step, baseline_rate=None):
    rows=[m for m in metrics if start_step<int(m['step'])<=end_step]
    if [m['step'] for m in rows] != list(range(start_step+1,end_step+1)):
        raise ValueError('Missing/duplicate qualification updates')
    for m in rows:
        if not all(math.isfinite(m.get(k,float('nan'))) for k in ('loss','grad_norm','valid_targets_per_second','max_rank_peak_allocated_gib')):
            raise ValueError('Nonfinite qualification metrics')
        if m['max_rank_peak_allocated_gib']>30.5 or m.get('sources/replay_targets')!=0:
            raise ValueError('Memory limit exceeded or unexpected replay')
        total=sum(m.get(f'sources/{s}/valid_targets',0) for s in ('edu','dclm'))
        # A source choice is one whole 8K row. Early updates cannot express
        # exactly 60/40; require the final cumulative mix within one point.
        if not total or abs(m['sources/edu/valid_targets']/total-.6)>max(.01,8192/total):
            raise ValueError('Token mixture differs from 60/40')
    final_total=sum(rows[-1][f'sources/{s}/valid_targets'] for s in ('edu','dclm'))
    if abs(rows[-1]['sources/edu/valid_targets']/final_total-.6)>.01:
        raise ValueError('Final qualification mixture differs from 60/40')
    rate=statistics.median(m['valid_targets_per_second'] for m in rows)
    if baseline_rate and rate<.75*baseline_rate:
        raise ValueError('Growth throughput below 75% of original median')
    return {'updates':len(rows),'median_valid_targets_per_second':rate}

def quality_gate(report, step):
    if report.get('step')!=step:
        raise ValueError('Validation does not describe the stopped checkpoint')
    values={'original':report['original'],**report['sources']}
    if set(values)!=set(report['baseline']):
        raise ValueError('Missing source baselines')
    for key,loss in values.items():
        base=report['baseline'][key]
        if not math.isfinite(loss) or not math.isfinite(base) or loss>base*1.01:
            raise ValueError(f'{key} validation regressed more than 1%; review before continuing')
    return values

def monitoring(settings, status):
    """Hand off CPU diagnostics and uploader; never modify the active 10B jobs."""
    if not settings.get('handoff_monitoring'):
        return
    root=Path(settings['data_dir']); url=status['wandb_url'].split('/')
    run_ref=f'{url[-4]}/{url[-3]}/{url[-1]}'
    wrapper=Path('/opt/supervisor-scripts/microllama-growth-quality.sh')
    # All names/settings are generated locally, never sourced from external text.
    import shlex
    q=shlex.quote
    wrapper.write_text('#!/bin/bash\nsource /opt/supervisor-scripts/utils/environment.sh\nsource /opt/supervisor-scripts/utils/logging.sh ""\nset -euo pipefail\ncd '+q(str(REPO))+'\nexport MICROLLAMA_DATA_ROOT='+q(str(root))+'\nexport CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2\nexec nice -n 10 '+q(settings['train_python'])+' -m scripts.evaluation.watch_checkpoints --train-pid '+str(status['pid'])+' --run '+q(settings['run_name'])+' --checkpoint-dir '+q(str(root/'checkpoints'))+' --wandb-run '+q(run_ref)+'\n')
    wrapper.chmod(0o755)
    subprocess.run(['supervisorctl','restart','microllama-growth-quality'],check=True,capture_output=True)
    config=root/'.hf-checkpoint-uploads/config.json'
    if config.exists():
        current=read(config); current.update(run_name=settings['run_name'],checkpoint_directory=str(root/'checkpoints'/settings['run_name']))
        atomic_json(config,current)

def tick(settings, state, launch=False):
    root=Path(settings['data_dir']); growth=Path(settings['growth_dir'])
    parent_logs=root/'logs'/settings['parent_run']; logs=root/'logs'/settings['run_name']
    parent=read(parent_logs/'status.json')
    stage=state.get('stage','waiting-for-10b')
    if stage in ('blocked','completed'):
        return state
    if parent.get('phase')=='failed' or parent.get('error') or parent.get('wandb_error'):
        raise RuntimeError('Parent training failed; growth cannot bypass it')
    preparation=growth/'prepare-status.json'
    if preparation.exists() and read(preparation).get('phase')=='failed':
        raise RuntimeError(f"Growth preparation failed: {read(preparation).get('error')}")
    catalog=growth/'catalog'
    if not (catalog/'manifest.json').exists():
        return {**state,'stage':'waiting-for-10b','waiting':'growth catalog not ready'}
    reader=MixtureTokenBlocks(catalog,8192)
    ready=reader.coverage(settings['target_tokens'])['balanced_unconsumed_targets']
    if stage=='waiting-for-10b':
        if parent.get('phase')!='completed' or parent.get('tokens',0)<10_000_000_000 or alive(parent.get('pid')):
            return {**state,'stage':stage,'waiting':'initial 10B still training','parent_tokens':parent.get('tokens'), 'prepared_balanced_targets':ready}
        healthy(parent,settings['gpus'])
        if parent['config']['target_tokens']!=10_000_000_000:
            raise ValueError('Parent is not the selected 10B stage')
        if ready<2_000_000_000:
            return {**state,'stage':stage,'waiting':'need 2B balanced admitted coverage','prepared_balanced_targets':ready}
        checkpoint=Path(parent['checkpoint']); bound_checkpoint(parent,settings)
        config=read(checkpoint/'run_config.json')
        if config.get('schedule')!='constant-after-warmup-v1' or config['sequence_length']!=8192 or config['learning_rate']!=.0003:
            raise ValueError('Unexpected parent context/held-rate recipe')
        config.update(run_name=settings['run_name'],target_tokens=settings['target_tokens'])
        old_metrics=[json.loads(line) for line in (parent_logs/'metrics.jsonl').read_text().splitlines()[-100:]]
        state.update(parent_checkpoint=str(checkpoint),parent_step=parent['step'],parent_tokens=parent['tokens'],
                     baseline_rate=statistics.median(m['valid_targets_per_second'] for m in old_metrics),config=config)
        return issue(settings,state,'operational',checkpoint,continue_from=True,launch=launch)
    status=read(logs/'status.json') if (logs/'status.json').exists() else {}
    if state.get('attempt'):
        if status.get('updated_unix',0)<state['attempt']['created_unix']:
            if time.time()-state['attempt']['created_unix']>600:
                raise RuntimeError('Launch outcome unconfirmed; refusing to launch again')
            return {**state,'waiting':'awaiting startup'}
        if alive(status.get('pid')):
            if not state['attempt'].get('monitors_handed_off'):
                if 'loss' not in status or 'validation_loss' not in status:
                    if time.time()-state['attempt']['created_unix']>600:
                        raise RuntimeError('Trainer did not reach a finite first update within ten minutes')
                    return {**state,'waiting':'initial model/validation loading'}
                healthy(status,settings['gpus'])
                if launch:
                    monitoring(settings,status)
                    state['attempt']['monitors_handed_off']=True
            return {**state,'waiting':'training','tokens':status.get('tokens'),'wandb_url':status.get('wandb_url')}
        if status.get('phase') not in ('stopped','completed','failed'):
            if time.time()-status.get('updated_unix',0)<600: return {**state,'waiting':'trainer exiting'}
            raise RuntimeError('Trainer disappeared without a complete checkpoint')
    healthy(status,settings['gpus'])
    if status.get('phase')=='completed':
        if status['tokens']<settings['target_tokens']: raise ValueError('Early completion')
        bound_checkpoint(status,settings)
        return {**state,'stage':'completed','tokens':status['tokens'],'checkpoint':status['checkpoint']}
    if stage=='full':
        raise RuntimeError('Full run stopped before target; explicit review/resume required')
    if status.get('phase')!='stopped' or not status.get('qualification_passed'):
        raise ValueError('Qualification did not stop successfully')
    if stage=='operational' and status['step']!=state['parent_step']+40:
        raise ValueError('Operational gate did not stop at 40 new updates')
    checkpoint=Path(status['checkpoint']); restart=bound_checkpoint(status,settings)
    if restart['data_mode']!='mixture-growth-v1' or restart['parent_tokens']!=state['parent_tokens']:
        raise ValueError('Qualification did not preserve the selected parent/source counters')
    metrics=[json.loads(line) for line in (logs/'metrics.jsonl').read_text().splitlines()]
    gate=metric_gate(metrics,state['parent_step'],status['step'],state['baseline_rate'])
    quality=quality_gate(read(logs/'growth-validation.json'),status['step'])
    state.setdefault('gates',{})[stage]={'checkpoint':str(checkpoint),'metrics':gate,'validation':quality,'tokens':status['tokens']}
    if stage=='operational':
        return issue(settings,state,'quality',checkpoint,max_new_tokens=250_000_000-(status['tokens']-state['parent_tokens']),launch=launch)
    if status['tokens']-state['parent_tokens']<250_000_000:
        raise ValueError('Quality pilot ended before 250M')
    return issue(settings,state,'full',checkpoint,launch=launch)

def issue(settings,state,stage,checkpoint,continue_from=False,max_new_tokens=None,launch=False):
    growth=Path(settings['growth_dir']); config=growth/'training-config.json'
    command=[settings['train_python'],'-m','scripts.cloud.run_vast','--hardware-profile',settings['hardware_profile'],
        '--gpus',str(settings['gpus']),'--data-dir',settings['data_dir'],'--token-dir',settings['legacy_token_dir'],
        '--growth-catalog',str(growth/'catalog'),'--config',str(config),'--run-name',settings['run_name'],'--engram',
        '--checkpoint-keep','4','--continue-from' if continue_from else '--resume',str(checkpoint)]
    if stage=='operational': command+=['--qualification-updates','40']
    if max_new_tokens: command+=['--max-new-tokens',str(max_new_tokens)]
    if not launch: return {**state,'ready_stage':stage,'planned_command':command}
    # Persist intent BEFORE launch. Unknown outcomes are blocked, never retried.
    atomic_json(config,state['config'])
    state.update(stage=stage,attempt={'created_unix':time.time(),'command':command,'monitors_handed_off':False})
    atomic_json(growth/'controller-state.json',state)
    env=dict(os.environ)
    private=Path(settings.get('private_environment','/nonexistent'))
    if private.exists(): env.update(read(private))
    result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True)
    (growth/f'launch-{stage}.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(f'{stage} launcher failed; see launch-{stage}.log')
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--inspect-checkpoint',type=Path)
    parser.add_argument('--tick',action='store_true',help='Allow already-authorized detached stage launches; otherwise readiness only')
    args=parser.parse_args()
    if args.inspect_checkpoint:
        print(json.dumps(checkpoint_report(args.inspect_checkpoint))); return
    if args.settings is None: parser.error('--settings required')
    settings=read(args.settings); growth=Path(settings['growth_dir'])
    lock=(growth/'.controller.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return
    path=growth/'controller-state.json'; state=read(path) if path.exists() else {}
    try: state=tick(settings,state,args.tick)
    except Exception as error:
        state.update(stage='blocked',error=f'{type(error).__name__}: {error}')
    state['updated_unix']=time.time(); atomic_json(path,state)
    print(json.dumps({k:v for k,v in state.items() if k not in ('config','attempt')}),flush=True)
    if state.get('stage')=='blocked': raise SystemExit(1)

if __name__=='__main__': main()
