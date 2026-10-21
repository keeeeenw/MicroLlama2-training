"""Distinct read-only-fixture full-model replay diagnostic; never production.

The original failed qualification and its checkpoint remain unchanged. This
tests the reviewed ordered embedding adapter on one real 8K microbatch, loading
the same saved model/Adam/data/RNG twice. It does not qualify global16 training.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from microllama.training import mac_run as runtime
from microllama.training.mac import update, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pilot, root = runtime.external(args.pilot), runtime.external(args.output)
    root.mkdir(exist_ok=False)
    launch = json.loads((pilot / 'launch.json').read_text())
    plan_path = Path(launch['command'][launch['command'].index('--plan') + 1])
    if sha256(plan_path) != launch['plan_sha256']:
        raise ValueError('Original pilot plan changed')
    plan = json.loads(plan_path.read_text())
    source_root = pilot / 'run/qualification/restart-fixture'
    pointer = json.loads((source_root / 'latest.json').read_text())
    checkpoint = source_root / pointer['checkpoint']
    if sha256(checkpoint / 'mac_manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Original qualification fixture pointer changed')
    result = {'kind': 'distinct-full-model-replay-diagnostic-v2',
              'original_failed_plan_sha256': launch['plan_sha256'],
              'fixture_manifest_sha256': pointer['manifest_sha256'],
              'implementation': runtime.source_bindings(), 'sequence_length': 8192,
              'microbatches': 1, 'production_training_started': False,
              'production_qualification_claimed': False, 'official_targets_advanced': 0,
              'new_download_bytes': 0, 'branches': []}
    def progress(value):
        write_json(root / 'status.json', {'unix': time.time(), **value})
    with runtime.device_lock(root.parent):
        torch.set_num_threads(2)
        torch.mps.set_per_process_memory_fraction(.8)
        branches = []
        for branch in range(2):
            progress({'phase': 'loading', 'branch': branch + 1})
            state = runtime.read_checkpoint(checkpoint, launch['plan_sha256'], kind='qualification-only')
            model, optimizer = runtime.restore_model(checkpoint, state, torch.device('mps'), True)
            del state['optimizer']
            reader, validator, stager = runtime.materialize(plan, state, 0)
            rows = reader.next_batch(1)
            initial = runtime.state_hashes(model, optimizer)
            rows_hash = hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()
            started = time.monotonic()
            measured = update(model, optimizer,
                [(torch.as_tensor(rows[:, :-1], device='mps'), torch.as_tensor(rows[:, 1:], device='mps'))],
                grad_clip=plan['config']['grad_clip'], chunk_size=plan['config']['loss_chunk_size'],
                progress=lambda value: progress({'phase': 'replaying', 'branch': branch + 1, **value}))
            branch_result = {'initial': initial, 'after': runtime.state_hashes(model, optimizer),
                             'rows_sha256': rows_hash, 'cursor': reader.state_dict(),
                             'rng': {key: value.clone() for key, value in runtime.rng_state(torch.device('mps')).items()}}
            branches.append(branch_result)
            result['branches'].append({**measured, 'seconds': time.monotonic() - started,
                                       'rows_sha256': rows_hash, 'after': branch_result['after']})
            del model, optimizer, reader, validator, state
            gc.collect()
            torch.mps.empty_cache()
        a, b = branches
        result['checks'] = {key: a[key] == b[key] for key in ('initial', 'rows_sha256', 'cursor')}
        result['checks'].update(model_exact=a['after']['model'] == b['after']['model'],
            adam_exact=a['after']['adam'] == b['after']['adam'],
            rng_exact=all(torch.equal(a['rng'][key], b['rng'][key]) for key in ('cpu', 'mps')))
        result['passed'] = all(result['checks'].values())
        write_json(root / 'result.json', result)
        progress({'phase': 'completed' if result['passed'] else 'failed', 'checks': result['checks']})
        print(json.dumps({'passed': result['passed'], 'checks': result['checks'],
                          'seconds': [b['seconds'] for b in result['branches']]}), flush=True)
        if not result['passed']:
            raise RuntimeError('Full-model reviewed replay still differs; preserve result, do not qualify')


if __name__ == '__main__':
    main()
