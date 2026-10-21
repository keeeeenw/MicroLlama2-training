"""Finite, disposable full8K speed experiments directly from original Vast50B.

No production entry point, checkpoint writer, tracking session or auto-start.
Variant substitutions exist only inside this diagnostic Python process.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import torch

from microllama.training import mac_mlx as graph
from microllama.training import mac_ultra_run as ultra
from microllama.training import mac_mlx_run as native
from microllama.training import mac_run as base
from microllama.training.mac import FINAL_MANIFEST, sha256, write_json


def select_variant(attention, embedding):
    if attention == 'native':
        graph.attention = lambda q, k, v, tile: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=q.shape[-1] ** -.5, mask='causal')
    if embedding == 'native':
        graph.ordered_take = lambda weight, ids: weight[ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attention', choices=['existing', 'native'], default='native')
    p.add_argument('--embedding', choices=['existing', 'native'], default='existing')
    p.add_argument('--batch', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--checkpointing', choices=['on', 'off'], default='on')
    p.add_argument('--loss-chunk', type=int, choices=[256, 512, 1024, 2048], default=256)
    p.add_argument('--trials', type=int, choices=[2, 4, 8, 16], default=2)
    p.add_argument('--cross-entropy', choices=['existing', 'fast'], default='existing')
    p.add_argument('--graph', choices=['experimental', 'fast'], default='experimental')
    p.add_argument('--accumulation-window', type=int, choices=[1, 2], default=1)
    p.add_argument('--fused-projections', action='store_true')
    p.add_argument('--ordered-gpu-embedding', action='store_true')
    p.add_argument('--cache-mib',type=int,choices=[512,2048,8192],default=512)
    args = p.parse_args()
    if args.graph == 'fast' and (args.embedding != 'existing' or args.attention != 'native'
                                or args.cross_entropy != 'fast'):
        p.error('Fast graph requires deterministic existing embedding, native attention and fast cross entropy')
    if args.graph != 'fast' and args.accumulation_window != 1:
        p.error('Evaluation overlap is implemented only by the clean fast graph')
    if args.graph != 'fast' and args.fused_projections:
        p.error('Projection fusion requires the clean fast graph')
    if args.graph != 'fast' and args.ordered_gpu_embedding:
        p.error('Ordered GPU reduction requires the clean fast graph')
    torch.set_num_threads(8)
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(64 * 2**30)
    mx.set_cache_limit(args.cache_mib * 2**20)
    root = base.external(args.output)
    root.mkdir(exist_ok=False)
    plan, old = ultra.load_plan(args.plan)
    if args.graph == 'experimental':
        select_variant(args.attention, args.embedding)
    if args.graph == 'experimental' and args.cross_entropy == 'fast':
        def cross_entropy(logits, targets, *, reduction='none'):
            if reduction != 'none':
                raise ValueError('This diagnostic preserves per-token masked loss')
            return mx.fast.cross_entropy(logits, targets)
        graph.nn.losses.cross_entropy = cross_entropy
    def report(value):
        item = {'unix': time.time(), **value}
        write_json(root / 'status.json', item)
        print(json.dumps(item), flush=True)
    with base.device_lock(Path(old['archive_cache']).parent):
        report({'phase': 'import-original-vast', 'attention': args.attention,
                'embedding': args.embedding, 'batch': args.batch, 'official_targets_advanced': 0})
        engine, state = ultra.import_vast(plan, args.batch)
        if args.graph == 'fast':
            from microllama.training.mac_mlx_fast import FastEngine
            engine = FastEngine(engine.params, engine.slots, engine.policy, engine.config,
                micro_batch_size=args.batch, chunk=args.loss_chunk,
                checkpoint=args.checkpointing == 'on', accumulation_window=args.accumulation_window,
                fused_projections=args.fused_projections,
                gpu_embedding=args.ordered_gpu_embedding,
                extras=engine.extras)
        else:
            engine.gradient = graph.scaled_value_and_grad(lambda p, batch, indices: graph.loss(
                p, engine.config, batch, indices, compute='bfloat16', query_tile=512,
                chunk=args.loss_chunk, checkpoint=args.checkpointing == 'on', reduction='sum'),
                compute='bfloat16', compiled=True, loss_scale=1)
        reader = base.stager_for(old, 0).reader(state['data'])
        rows = reader.next_batch(16)
        original_params, original_slots = engine.params, engine.slots
        measurements = []
        for trial in range(args.trials):
            engine.params, engine.slots = original_params, original_slots
            mx.eval(engine.params, engine.slots)
            measured = engine.update(rows, lambda v: report({'phase': 'speed-trial', 'trial': trial, **v}))
            measurements.append(measured)
            report({'phase': 'trial-complete', 'trial': trial, **measured})
        engine.params, engine.slots = original_params, original_slots
        mx.eval(engine.params, engine.slots)
        numerical = None
        if args.oracle:
            from microllama.training.mac_gradient_check import full_gradient_check
            oracle = json.loads((args.oracle / 'result.json').read_text())
            one = rows[:1]
            assert hashlib.sha256(one.astype('<u2').tobytes()).hexdigest() == oracle['rows_sha256']
            count = int(((one[:, :-1] != 7) & (one[:, :-1] != 0) & (one[:, 1:] != 0)).sum())
            value, gradients = engine.gradient(engine.params, mx.array(one), engine.indices(one))
            gradients = {n: g/count for n, g in gradients.items()}
            mx.eval(value, gradients)
            comparison = full_gradient_check(gradients, reference=args.oracle / 'gradients')
            delta = abs(float(value)/count-oracle['loss'])
            hashes = {n: native.array_hash(gradients[n]) for n in sorted(gradients)}
            del gradients
            value2, gradients2 = engine.gradient(engine.params, mx.array(one), engine.indices(one))
            gradients2 = {n: g/count for n, g in gradients2.items()}
            mx.eval(value2, gradients2)
            repeats = {n: native.array_hash(gradients2[n]) for n in sorted(gradients2)}
            exact = hashes == repeats
            numerical = {'comparison': comparison, 'absolute_loss_difference': delta,
                         'repeated_gradient_exact': exact,
                         'nonrepeatable_parameter_names': [n for n in hashes if hashes[n] != repeats[n]],
                         'passed': comparison['cosine']>.999 and comparison['relative_l2']<.05 and delta<.005}
        result = {'kind': 'disposable-speed-experiment', 'official_targets_advanced': 0,
                  'production_started': False, 'no_production_autostart': True,
                  'parent_manifest_sha256': FINAL_MANIFEST, 'plan_sha256': sha256(args.plan),
                  'experiment_source_sha256': sha256(Path(__file__)),
                  'attention': args.attention, 'embedding': args.embedding,
                  'activation_checkpointing': args.checkpointing, 'loss_chunk': args.loss_chunk,
                  'cross_entropy': args.cross_entropy,
                  'graph': args.graph, 'accumulation_window': args.accumulation_window,
                  'fused_projections': args.fused_projections,
                  'ordered_gpu_embedding': args.ordered_gpu_embedding,
                  'cache_limit_bytes':args.cache_mib*2**20,
                  'gpu_embedding_sources': {str(p.name):sha256(p) for p in [Path('microllama/training/mac_gpu_embedding.py'),Path('microllama/training/metal/embedding_scatter.metal')]} if args.ordered_gpu_embedding else None,
                  'fast_graph_sha256': sha256(Path(__file__).resolve().parents[2]/'microllama/training/mac_mlx_fast.py') if args.graph == 'fast' else None,
                  'backend_environment': {n: os.environ[n] for n in ('MLX_SDPA_VJP_BLOCK','MLX_SDPA_VJP_BQ','MLX_SDPA_VJP_BK','MLX_ENABLE_TF32','MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER') if n in os.environ},
                  'physical_batch': args.batch, 'sequence_length': 8192, 'global_batch': 16,
                  'measurements': measurements, 'numerical': numerical}
        write_json(root / 'result.json', result)
        report({'phase': 'completed', 'valid_targets_per_second': measurements[-1]['valid_targets_per_second'],
                'numerical_passed': numerical['passed'] if numerical else None, 'production_started': False})


if __name__ == '__main__':
    main()
