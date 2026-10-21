"""Compare the unchanged saved next row with a saved CPU gradient oracle; no updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import mlx.core as mx

from microllama.training.mac import sha256, write_json
from microllama.training.mac_run import external, stager_for, device_lock
from microllama.training.mac_mlx import loss, memory_indices, scaled_value_and_grad
from microllama.training.mac_gradient_check import full_gradient_check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint', 'plan', 'oracle', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--compute', choices=['float32', 'bfloat16'])
    args = p.parse_args()
    root = external(args.output); root.mkdir(exist_ok=False)
    oracle = json.loads((args.oracle / 'results.json').read_text())
    if (sha256(args.checkpoint / 'mac_manifest.json') != oracle['parent_manifest_sha256']
            or sha256(args.plan) != oracle['parent_plan_sha256']):
        raise ValueError('Oracle parent differs')
    plan = json.loads(args.plan.read_text())
    saved = torch.load(args.checkpoint / 'mac_state.pt', map_location='cpu', weights_only=True, mmap=True)
    rows = stager_for(plan, 0).reader(saved['data']).next_batch(1)[:, :oracle['length'] + 1]
    del saved
    if hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest() != oracle['rows_sha256']:
        raise ValueError('Oracle row differs')
    with device_lock(root.parent):
        mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
        mx.set_memory_limit(18 * 2**30); mx.set_cache_limit(512 * 2**20)
        loaded = mx.load(str(args.checkpoint / 'model.safetensors'))
        # Use the exact independently enumerated trainable names, excluding
        # checkpoint buffers (including the integer Engram canonical map).
        names = json.loads((args.oracle / 'cpu-reference/manifest.json').read_text())
        params = {n: loaded[n] for n in names}
        del loaded
        config = json.loads((args.checkpoint / 'config.json').read_text())
        sizes = [params[f'model.engram.{order}_tables.{h}.weight'].shape[0]
                 for order in ['bigram', 'trigram'] for h in range(config['engram_hash_heads'])]
        indices = memory_indices(config, rows[:, :-1], sizes)
        count = int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
        compute = args.compute or oracle['precision']
        fn = scaled_value_and_grad(lambda w, b, i: loss(w, config, b, i, compute=compute,
            query_tile=oracle['query_tile'], chunk=oracle['loss_chunk'], checkpoint=False, reduction='sum'),
            compute=compute, compiled=not args.no_compile, loss_scale=1)
        start = time.monotonic(); value, gradients = fn(params, mx.array(rows), indices)
        gradients = {n: g / count for n, g in gradients.items()}
        norm = mx.sqrt(sum(mx.sum(gradients[n] ** 2) for n in sorted(gradients)))
        gradients = {n: g * mx.minimum(1., 1. / (norm + 1e-6)) for n, g in gradients.items()}
        mx.eval(value, gradients, norm)
        result = {'kind': 'diagnostic-only', 'official_targets_advanced': 0,
                  'device': args.device, 'compiled': not args.no_compile, 'precision': compute,
                  'oracle_precision': oracle['precision'],
                  'loss': float(value) / count, 'gradient_norm': float(norm),
                  'seconds': time.monotonic() - start, 'oracle_results_sha256': sha256(args.oracle / 'results.json'),
                  'implementation_sha256': sha256(Path(__file__).parents[2] / 'microllama/training/mac_mlx.py'),
                  'comparison': full_gradient_check(gradients, reference=args.oracle / 'cpu-reference')}
        write_json(root / 'result.json', result)
        print(json.dumps({**{k: v for k, v in result.items() if k != 'comparison'},
                          'comparison': {k: v for k, v in result['comparison'].items() if k != 'per_parameter'}}), flush=True)


if __name__ == '__main__':
    main()
