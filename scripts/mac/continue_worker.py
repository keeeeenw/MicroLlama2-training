"""One reviewed budget extension after a healthy, source-frozen Mac pilot.

Waits without GPU work. Never retries a failed pilot or starts a second trainer.
The existing qualification remains the proof for the unchanged implementation;
only the execution budget changes, with a separate recorded authorization.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def healthy_budget_end(status, launch, old_plan, *, now):
    """A user stop, missing outcome or failed attempt must not trigger continuation."""
    phase = status.get('phase')
    if phase == 'failed':
        raise RuntimeError('Pilot failed; preserving it without automatic retry')
    if phase not in ('completed', 'paused'):
        raise RuntimeError('Pilot exited without a healthy completed budget outcome')
    if type(status.get('updates')) is not int or status['updates'] < 1:
        raise RuntimeError('No complete production update exists')
    if not isinstance(status.get('checkpoint'), str):
        raise RuntimeError('No complete production checkpoint exists')
    if phase == 'completed':
        if not old_plan['config']['target_tokens'] <= status['tokens'] < old_plan['config']['target_tokens'] + 131072:
            raise RuntimeError('Pilot completion counters differ from its finite target')
    elif (now - launch['unix'] < launch['max_seconds']
          and status['updates'] < launch['max_updates']):
        raise RuntimeError('Pilot paused before its budget; treating this as a user stop')


def extend_plan(plan, catalog):
    result = copy.deepcopy(plan)
    # Existing catalog and data capacity remain hard safety bounds, not new data.
    target = min(catalog['target_tokens'], plan['parent_tokens'] +
                 plan['coverage']['balanced_unconsumed_targets'] - 131072)
    if type(target) is not int or target <= plan['config']['target_tokens']:
        raise ValueError('No already-admitted dataset headroom for continuation')
    result['config']['target_tokens'] = target
    result['target_authorization'] = 'Direct human: Keep running until I stop it; existing data/transfer/quality gates retained'
    return result


def check_transition(old_plan, new_plan, state):
    expected = copy.deepcopy(old_plan)
    expected['config']['target_tokens'] = new_plan['config']['target_tokens']
    expected['target_authorization'] = new_plan['target_authorization']
    if expected != new_plan or state['config'] != old_plan['config'] or state['validation'] != old_plan['validation']:
        raise ValueError('Budget transition changed a qualified recipe or checkpoint configuration')
    if not old_plan['parent_tokens'] < state['tokens'] < new_plan['config']['target_tokens']:
        raise ValueError('Continuation counters are outside the admitted dataset budget')


def remaining_download_bytes(launch, qualification, events):
    used_training = 0
    for event in events:
        if 'download_bytes_requested' in event:
            value = event['download_bytes_requested']
            if type(value) is not int or value < used_training:
                raise ValueError('Invalid training transfer accounting')
            used_training = value
    qualified = qualification['download_bytes_requested']
    if type(qualified) is not int or qualified < 0:
        raise ValueError('Invalid qualification transfer accounting')
    remaining = launch['max_download_bytes'] - qualified - used_training
    if remaining < 0:
        raise ValueError('Original combined transfer allowance was exceeded')
    return remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    # Import only after the launcher sets PYTHONPATH to the already-frozen code.
    from microllama.training import mac_run as runtime
    from microllama.training.mac import sha256, write_json
    from microllama.training.mac_tracking import Tracker, OLD_CUDA_RUN
    import torch

    pilot, root = runtime.external(args.pilot), runtime.external(args.output)
    root.mkdir(parents=True, exist_ok=False)
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        write_json(root / 'stop-requested.json', {'unix': time.time(), 'signal': signum})
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def report(value):
        value = {'unix': time.time(), 'pid': os.getpid(), **value}
        write_json(root / 'status.json', value)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n'); stream.flush(); os.fsync(stream.fileno())

    try:
        launch = read(pilot / 'launch.json')
        if launch['plan_sha256'] != args.expected_plan_sha256:
            raise ValueError('Unexpected pilot launch plan')
        if Path(runtime.__file__).resolve().parents[2] != Path(launch['source_snapshot']).resolve():
            raise ValueError('Continuation must import the pilot source snapshot')
        command = launch['command']
        old_path = Path(command[command.index('--plan') + 1])
        if sha256(old_path) != args.expected_plan_sha256:
            raise ValueError('Original pilot plan changed')
        old_plan = read(old_path)
        if runtime.source_bindings() != old_plan['implementation']:
            raise ValueError('Frozen qualified implementation changed')
        report({'phase': 'waiting-for-healthy-pilot-budget', 'pilot_pid': launch['pid'],
                'policy': 'until-user-stop', 'additional_transfer_allowance': 0,
                'no_automatic_failure_retry': True})
        while not stopped and not (root / 'STOP').exists():
            try:
                os.kill(launch['pid'], 0)
            except ProcessLookupError:
                break
            time.sleep(5)
        if stopped or (root / 'STOP').exists():
            report({'phase': 'cancelled-before-handoff'})
            return
        outcome = read(pilot / 'run/worker-status.json')
        healthy_budget_end(outcome, launch, old_plan, now=time.time())
        # A second observer cannot duplicate this work: the same device lock
        # protects qualification, original training and the complete handoff.
        with runtime.device_lock(Path(old_plan['archive_cache']).parent):
            runtime.load_plan(old_path)
            qualified_path = pilot / 'run/qualification/qualification.json'
            qualified = read(qualified_path)
            runtime.require_qualification(qualified, args.expected_plan_sha256, runtime.hardware())
            if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
                raise RuntimeError('MPS unavailable or blanket fallback enabled')
            if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') in ('0', '0.0'):
                raise RuntimeError('Allocator safety limit disabled')
            torch.mps.set_per_process_memory_fraction(old_plan['mps_memory_fraction'])
            torch.set_num_threads(2)
            new_plan = extend_plan(old_plan, runtime.load_catalog(old_plan['catalog']))
            plan_path = root / 'plan.json'
            write_json(plan_path, new_plan)
            plan_sha = sha256(plan_path)
            runtime.load_plan(plan_path)
            checkpoint = runtime.external(outcome['checkpoint'])
            if checkpoint.parent != pilot / 'run/training/checkpoints':
                raise ValueError('Pilot checkpoint is outside its production directory')
            latest = read(checkpoint.parent / 'latest.json')
            if (latest['checkpoint'] != checkpoint.name or
                    latest['manifest_sha256'] != sha256(checkpoint / 'mac_manifest.json')):
                raise ValueError('Pilot checkpoint differs from its published latest pointer')
            state = runtime.read_checkpoint(checkpoint, args.expected_plan_sha256)
            if state['tokens'] != outcome['tokens'] or state['step'] != outcome['step']:
                raise ValueError('Pilot status and checkpoint counters differ')
            check_transition(old_plan, new_plan, state)
            transfer_events = [json.loads(line) for line in
                               (pilot / 'run/training/events.jsonl').read_text().splitlines()]
            allowance = remaining_download_bytes(launch, qualified, transfer_events)
            write_json(root / 'budget-extension.json', {
                'unix': time.time(), 'authorization': new_plan['target_authorization'],
                'qualification_repeated': False, 'qualification_sha256': sha256(qualified_path),
                'qualification_plan_sha256': args.expected_plan_sha256,
                'plan_sha256': plan_sha, 'qualified_implementation': old_plan['implementation'],
                'entry_checkpoint': str(checkpoint), 'entry_manifest_sha256': latest['manifest_sha256'],
                'previous_target': old_plan['config']['target_tokens'],
                'dataset_safety_target': new_plan['config']['target_tokens'],
                'wall_clock_limit': None, 'pilot_token_limit_removed': True,
                'remaining_original_download_bytes': allowance,
                'controller_sha256': sha256(Path(__file__)), 'no_autorestart': True})
            state['config'] = copy.deepcopy(new_plan['config'])
            model, optimizer = runtime.restore_model(checkpoint, state, torch.device('mps'),
                                                       new_plan['config']['gradient_checkpointing'])
            del state['optimizer']
            reader, validator, stager = runtime.materialize(new_plan, state, allowance)
            reader.coverage(new_plan['config']['target_tokens'])
            previous_identity = read(pilot / 'run/training/tracking/identity.json')
            if (previous_identity['plan_sha256'] != args.expected_plan_sha256
                    or previous_identity['id'] == OLD_CUDA_RUN):
                raise ValueError('Unexpected tracking identity')
            tracking_dir = root / 'tracking'
            tracking_dir.mkdir()
            write_json(tracking_dir / 'identity.json', {**previous_identity, 'plan_sha256': plan_sha,
                'previous_plan_sha256': args.expected_plan_sha256, 'budget_extension': True})
            tracker = Tracker(tracking_dir, plan_sha, {'plan_sha256': plan_sha,
                'config': new_plan['config'], 'kind': 'authorized-mac-budget-extension'})
            def progress(value):
                value = {**value, 'wandb_id': tracker.identity['id'],
                         'wandb_mode': tracker.mode, 'download_bytes_requested': stager.download_bytes}
                report(value); tracker.log(value)
            loop = runtime.TrainingLoop(model, optimizer, reader, validator, state, new_plan['config'],
                validation_reference={key: value['loss'] for key, value in qualified['baseline'].items()},
                max_validation_increase=new_plan['max_validation_increase'],
                save=lambda m, o, s: runtime.publish(root / 'checkpoints', m, o, s,
                    plan_sha=plan_sha, assets_from=new_plan['parent']), report=progress,
                initial_checkpoint=str(checkpoint))
            try:
                # Every update has >=1 valid target. These numerical ceilings
                # therefore cannot precede the existing dataset safety endpoint;
                # no 24-hour/10M pilot ceiling is imposed on this continuation.
                result = loop.run(max_updates=new_plan['config']['target_tokens'] - state['tokens'] + 1,
                    max_seconds=sys.float_info.max,
                    stopped=lambda: stopped or (root / 'STOP').exists())
                progress({**result, 'policy': 'until-user-stop-or-safety-gate'})
            except BaseException:
                tracker.finish(exit_code=1)
                raise
            else:
                tracker.finish()
    except BaseException as error:
        report({'phase': 'failed', 'error': f'{type(error).__name__}: {error}', 'autorestart': False})
        raise


if __name__ == '__main__':
    main()
