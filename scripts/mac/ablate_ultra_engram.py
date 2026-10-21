"""Paired finite Engram ablation from Vast50B; never publishes or trains onward."""
import argparse
import copy
import json
from pathlib import Path
import time

import mlx.core as mx
import torch

from microllama.training import mac_ultra_run as ultra, mac_run as base, mac_mlx_run as native
from microllama.training.mac_mlx_fast import FastEngine
from microllama.training.mac import FINAL_MANIFEST, sha256, write_json


def without_engram(original, *, batch=1):
    config = copy.deepcopy(original.config)
    config['engram_layers'] = []
    engine = FastEngine(original.params, original.slots, original.policy, config,
                        micro_batch_size=batch, extras=original.extras)
    # The ablation removes the parameters and their optimizer work as well as
    # the residual branch. It is a different model, never a continuation export.
    engine.params = {n: v for n, v in engine.params.items() if not n.startswith('model.engram.')}
    engine.slots = {n: v for n, v in engine.slots.items() if n in engine.params}
    engine.policy = {n: v for n, v in engine.policy.items() if n in engine.params}
    engine.indices = lambda rows: (mx.array(0),)*4
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--validation-blocks', type=int, choices=[0, 8], default=8)
    args = parser.parse_args()
    torch.set_num_threads(8)
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(64*2**30)
    mx.set_cache_limit(512*2**20)
    plan, old = ultra.load_plan(args.plan)
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    def report(item):
        item = {'unix': time.time(), **item}
        write_json(root/'status.json', item)
        print(json.dumps(item), flush=True)
    with base.device_lock(Path(old['archive_cache']).parent):
        original, state = ultra.import_vast(plan, args.batch)
        stager = base.stager_for(old, 0)
        rows = stager.reader(state['data']).next_batch(16)
        result = {'kind': 'engram-architecture-ablation', 'production_started': False,
                  'official_targets_advanced': 0, 'parent_manifest_sha256': FINAL_MANIFEST,
                  'plan_sha256': sha256(args.plan), 'sequence_length': 8192,
                  'global_batch': 16, 'physical_batch': args.batch,
                  'source_sha256': sha256(Path(__file__)),
                  'fast_graph_sha256': sha256(Path(__file__).resolve().parents[2]/'microllama/training/mac_mlx_fast.py'),
                  'variants': {}}
        for enabled in (True, False):
            name = 'engram-on' if enabled else 'engram-off'
            engine = (FastEngine(original.params, original.slots, original.policy, original.config,
                                 micro_batch_size=args.batch, extras=original.extras)
                      if enabled else without_engram(original, batch=args.batch))
            initial_params, initial_slots = engine.params, engine.slots
            measurements = []
            for trial in range(2):
                engine.params, engine.slots = initial_params, initial_slots
                mx.eval(engine.params, engine.slots)
                measurements.append(engine.update(rows, lambda item: report({'variant': name, 'trial': trial, **item})))
            # Measure heldout loss at imported weights, discarding both updates.
            engine.params, engine.slots = initial_params, initial_slots
            validation = None
            if args.validation_blocks:
                spec = copy.deepcopy(plan['validation'])
                spec.update(blocks=args.validation_blocks, minimum_valid_targets=1)
                validator = native.Validation(spec, stager)
                validator.progress = lambda item: report({'variant': name, **item})
                validation = validator(engine)
            result['variants'][name] = {'measurements': measurements, 'validation_screen': validation,
                'active_parameter_elements': sum(p.size for p in engine.params.values())}
            write_json(root/'result.json', result)
            report({'phase': 'variant-complete', 'variant': name,
                    'speed': measurements[-1]['valid_targets_per_second']})
        a, b = (result['variants'][name] for name in ('engram-on', 'engram-off'))
        result['speed_ratio_without_engram'] = b['measurements'][-1]['valid_targets_per_second']/a['measurements'][-1]['valid_targets_per_second']
        if args.validation_blocks:
            result['heldout_loss_increase'] = {s: b['validation_screen'][s]['loss']-a['validation_screen'][s]['loss']
                                               for s in a['validation_screen']}
        write_json(root/'result.json', result)
        report({'phase': 'completed', 'speed_ratio_without_engram': result['speed_ratio_without_engram'],
                'heldout_loss_increase': result.get('heldout_loss_increase'), 'production_started': False})


if __name__ == '__main__':
    main()
