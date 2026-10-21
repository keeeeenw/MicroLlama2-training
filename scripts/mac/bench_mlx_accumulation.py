"""Finite real-global16 MLX timing; discard the diagnostic optimizer update.

Requires the completed matched speed comparison and its numerical screen.
Never writes production checkpoints, cursors, W&B or official counters.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from microllama.training import mac_run as runtime
from microllama.training.mac import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['float16', 'bfloat16'], default='bfloat16')
    args = parser.parse_args()
    root = runtime.external(args.output)
    root.mkdir(exist_ok=False)
    def report(**value):
        write_json(root / 'status.json', {'unix': time.time(), **value})
    result = {'kind': 'finite-mlx-global16-training-diagnostic',
              'official_targets_advanced': 0, 'new_downloads': 0,
              'training_resumed': False, 'production_qualified': False}
    try:
        comparison = runtime.external(args.comparison)
        summary_path = comparison / 'summary.json'
        summary = json.loads(summary_path.read_text())
        candidate = summary['comparisons'][{'float16': 'mlx-fp16-tiled', 'bfloat16': 'mlx-bf16-tiled'}[args.precision]]
        if not candidate['experimental_numerical_screen_passed']:
            raise ValueError('Matched loss/gradient screen must pass before global16 timing')
        path = Path(summary['checkpoint'])
        result.update(checkpoint=str(path), parent_manifest_sha256=sha256(path / 'mac_manifest.json'),
                      comparison_sha256=sha256(summary_path))
        with runtime.device_lock(Path(args.plan).parent):
            report(phase='loading')
            import mlx.core as mx
            from microllama.training.mac_mlx import (adam_from_torch, adam_update,
                loss, memory_indices, scaled_value_and_grad)
            from microllama.training.engram import EngramQwen3Config, EngramQwen3ForCausalLM
            from microllama.training.optimizers import build_optimizer
            mx.set_default_device(mx.gpu)
            # Global16 holds one extra FP32 gradient accumulator. Keep bounded
            # headroom for it on the 32GB unified-memory machine.
            mx.set_memory_limit(18 * 2**30)
            mx.set_cache_limit(512 * 2**20)
            plan = json.loads(args.plan.read_text())
            state = runtime.read_checkpoint(path, sha256(args.plan))
            rows = runtime.stager_for(plan, 0).reader(state['data']).next_batch(16)
            if rows.shape != (16, 8193):
                raise ValueError('Require original global16, full8K rows')
            result['rows_sha256'] = hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()
            counts = ((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum(axis=1)
            total = int(counts.sum())
            config = json.loads((path / 'config.json').read_text())
            with torch.device('meta'):
                model = EngramQwen3ForCausalLM(EngramQwen3Config.from_pretrained(path, local_files_only=True))
            optimizer = build_optimizer(model, fused=False, memory_lr_multiplier=5)
            slots, policy = adam_from_torch(model, optimizer, state['optimizer'])
            names = {name for name, _ in model.named_parameters()}
            loaded = mx.load(str(path / 'model.safetensors'))
            params = {name: loaded[name].astype(mx.float32) for name in names}
            del loaded, state, optimizer, model
            gc.collect()
            sizes = [params[f'model.engram.{order}_tables.{h}.weight'].shape[0]
                     for order in ['bigram', 'trigram'] for h in range(config['engram_hash_heads'])]
            fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices,
                compute=args.precision, query_tile=512, chunk=1024, checkpoint=True),
                compute=args.precision, compiled=True)
            batches = [(mx.array(row[None]), memory_indices(config, row[None, :-1], sizes)) for row in rows]
            mx.eval(params, slots, batches)
            report(phase='warmup')
            value, gradients = fn(params, *batches[0])
            mx.eval(value, gradients)
            del value, gradients
            mx.reset_peak_memory()
            started = time.perf_counter()
            accumulated = None
            losses, timings = [], []
            for index, (batch, indices) in enumerate(batches):
                micro_start = time.perf_counter()
                value, gradients = fn(params, batch, indices)
                # Each micro loss is a valid-target mean. Weight before
                # accumulating to recover the original global token mean.
                weight = float(counts[index]) / total
                accumulated = {name: g * weight + (0 if accumulated is None else accumulated[name])
                               for name, g in gradients.items()}
                mx.eval(value, accumulated)
                micro_seconds = time.perf_counter() - micro_start
                losses.append(float(value)); timings.append(micro_seconds)
                if not np.isfinite(losses[-1]):
                    raise RuntimeError('Nonfinite microbatch loss')
                del gradients, value
                report(phase='accumulating', completed_microbatches=index + 1, total_microbatches=16,
                       last_micro_seconds=micro_seconds)
            optimizer_start = time.perf_counter()
            next_params, next_slots, norm = adam_update(params, accumulated, slots, policy)
            mx.eval(next_params, next_slots, norm)
            finished = time.perf_counter()
            if not np.isfinite(float(norm)):
                raise RuntimeError('Nonfinite accumulated gradient norm')
            result.update(completed=True, valid_targets=total, micro_valid_targets=counts.tolist(),
                micro_losses=losses, micro_seconds=timings, seconds=finished - started,
                optimizer_seconds=finished - optimizer_start,
                valid_targets_per_second=total / (finished - started),
                loss=float(np.dot(losses, counts) / total), gradient_norm=float(norm),
                peak_bytes=mx.get_peak_memory(), memory_limit_bytes=18 * 2**30,
                compute=args.precision, masters_and_moments='float32', loss_scale=8192 if args.precision == 'float16' else 1,
                query_tile=512, loss_chunk=1024, compiled=True,
                discarded_optimizer_update=True)
            write_json(root / 'result.json', result)
            report(phase='completed', valid_targets_per_second=result['valid_targets_per_second'])
    except BaseException as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
        write_json(root / 'result.json', result)
        report(phase='failed', error=result['error'])
        raise


if __name__ == '__main__':
    main()
