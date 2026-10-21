"""Matched 8K forward/backward/Adam trials; never continues production.

Each case starts from the same complete Mac checkpoint and next reader row.
Warmup gradients are discarded. No checkpoint, W&B run, cursor or official
counter is written. All outputs/caches stay on TrainData. Run after pausing.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from microllama.training import mac_run as runtime
from microllama.training.mac import write_json, sha256


def gradient_sample_indices(size):
    if size < 1:
        raise ValueError('Cannot sample an empty parameter')
    # Float32 linspace can round size-1 up to size for large embeddings.
    # Use the identical, in-bounds integer positions on both backends.
    return np.arange(8, dtype=np.int64) * (size - 1) // 7


def sample_gradients(values):
    result = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            indices = torch.as_tensor(gradient_sample_indices(value.numel()), device=value.device)
            result[name] = value.detach().reshape(-1)[indices].float().cpu().tolist()
        else:
            import mlx.core as mx
            indices = mx.array(gradient_sample_indices(value.size).astype(np.int32))
            result[name] = np.array(value.reshape(-1)[indices]).tolist()
    return result


def gradient_evidence(args, values):
    result = {'gradient_samples': sample_gradients(values)}
    if args.export_gradients or args.reference_gradients:
        from microllama.training.mac_gradient_check import full_gradient_check
        result['full_gradient_check'] = full_gradient_check(values, export=args.export_gradients,
                                                          reference=args.reference_gradients)
    return result


def torch_case(args, path, state, rows, plan, report):
    from microllama.training.mac_attention import install_tiled_attention
    from microllama.training.objective import hidden_nll, target_mask, normalize_accumulated_gradients
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.8)
    model, optimizer = runtime.restore_model(path, state, torch.device('mps'), True)
    del state['optimizer']
    install_tiled_attention(model, args.tile)
    if args.precision != 'float32':
        # This Transformers version disables CPU, rather than MPS, autocast
        # inside RoPE. Keep long-position phase arithmetic explicitly FP32.
        original_rotary = model.model.rotary_emb.forward
        def rotary(self, *values, **kwargs):
            with torch.autocast('mps', enabled=False):
                return original_rotary(*values, **kwargs)
        model.model.rotary_emb.forward = types.MethodType(rotary, model.model.rotary_emb)
    x, y = [torch.as_tensor(value, device='mps') for value in (rows[:, :-1], rows[:, 1:])]
    mask = target_mask(x, y, 7, 0)
    count = int(mask.sum().cpu())
    measured = []
    for trial in range(2):
        report({'phase': 'warmup' if trial == 0 else 'measuring'})
        model.train(); optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize(); start = time.perf_counter()
        with torch.autocast('mps', dtype=torch.float16 if args.precision == 'float16' else torch.bfloat16,
                            enabled=args.precision != 'float32'):
            hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
            torch.mps.synchronize(); backbone_end = time.perf_counter()
            nll = hidden_nll(hidden, model.lm_head.weight, y, args.chunk, mask)
            torch.mps.synchronize(); loss_end = time.perf_counter()
        if not bool(torch.isfinite(nll)):
            raise RuntimeError('Nonfinite trial loss')
        nll.backward()
        torch.mps.synchronize(); backward_end = time.perf_counter()
        normalize_accumulated_gradients(model, count)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if trial:
            optimizer.step()
        torch.mps.synchronize(); end = time.perf_counter()
        measured.append({'loss': float(nll.detach().cpu()) / count, 'valid_targets': count,
            'seconds': end - start, 'backbone_forward_seconds': backbone_end - start,
            'loss_forward_seconds': loss_end - backbone_end,
            'backward_seconds': backward_end - loss_end, 'optimizer_seconds': end - backward_end,
            'grad_norm': float(norm.cpu()), 'driver_bytes_at_update_end': torch.mps.driver_allocated_memory()})
        del hidden, nll
    measured[-1].update(gradient_evidence(args, {name: p.grad for name, p in model.named_parameters()}))
    return measured


def mlx_case(args, path, state, rows, plan, report):
    import mlx.core as mx
    from microllama.training.mac_mlx import memory_indices, loss, adam_from_torch, adam_update, scaled_value_and_grad
    from microllama.training.engram import EngramQwen3Config, EngramQwen3ForCausalLM
    from microllama.training.optimizers import build_optimizer
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(14 * 2**30)
    mx.set_cache_limit(512 * 2**20)
    config = json.loads((path / 'config.json').read_text())
    with torch.device('meta'):
        metadata_model = EngramQwen3ForCausalLM(EngramQwen3Config.from_pretrained(path, local_files_only=True))
    metadata_optimizer = build_optimizer(metadata_model, fused=False, memory_lr_multiplier=5)
    slots, policy = adam_from_torch(metadata_model, metadata_optimizer, state['optimizer'])
    names = {name for name, _ in metadata_model.named_parameters()}
    loaded = mx.load(str(path / 'model.safetensors'))
    params = {name: loaded[name].astype(mx.float32) for name in names}
    del loaded, state['optimizer'], metadata_optimizer, metadata_model
    gc.collect()
    sizes = [params[f'model.engram.{order}_tables.{h}.weight'].shape[0]
             for order in ['bigram', 'trigram'] for h in range(config['engram_hash_heads'])]
    indices = memory_indices(config, rows[:, :-1], sizes)
    batch = mx.array(rows)
    count = int(((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum())
    fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices, compute=args.precision,
        query_tile=args.tile, chunk=args.chunk, checkpoint=True, reduction='sum'),
        compute=args.precision, compiled=args.compiled, loss_scale=1)
    mx.eval(params, slots, batch, indices)
    measured = []
    for trial in range(2):
        report({'phase': 'warmup' if trial == 0 else 'measuring'})
        mx.reset_peak_memory()
        start = time.perf_counter()
        value, gradients = fn(params, batch, indices)
        # Match CUDA: backprop summed NLL, then normalize FP32 gradients.
        value = value / count
        gradients = {name: g / count for name, g in gradients.items()}
        mx.eval(value, gradients)
        grad_end = time.perf_counter()
        next_params, next_slots, norm = adam_update(params, gradients, slots, policy)
        mx.eval(next_params, next_slots, norm)
        end = time.perf_counter()
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise RuntimeError('Nonfinite MLX loss or gradient norm')
        measured.append({'loss': float(value), 'valid_targets': count, 'seconds': end - start,
            'forward_backward_seconds': grad_end - start, 'optimizer_seconds': end - grad_end,
            'grad_norm': float(norm), 'peak_bytes': mx.get_peak_memory()})
        if trial:
            scale = mx.minimum(mx.array(1., mx.float32), 1. / (norm + 1e-6))
            measured[-1].update(gradient_evidence(args, {name: value * scale for name, value in gradients.items()}))
        del next_params, next_slots, gradients
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['torch', 'mlx'], required=True)
    parser.add_argument('--precision', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--compiled', action='store_true')
    gradient_options = parser.add_mutually_exclusive_group()
    gradient_options.add_argument('--export-gradients', type=Path)
    gradient_options.add_argument('--reference-gradients', type=Path)
    args = parser.parse_args()
    if args.export_gradients:
        args.export_gradients = runtime.external(args.export_gradients)
    if args.reference_gradients:
        args.reference_gradients = runtime.external(args.reference_gradients)
    root, path = runtime.external(args.output), runtime.external(args.checkpoint)
    root.mkdir(exist_ok=False)
    def report(value):
        write_json(root / 'status.json', {'unix': time.time(), **value})
    result = {'kind': '8k-one-microbatch-training-speed-trial', 'backend': args.backend,
        'precision': args.precision, 'query_tile': args.tile, 'loss_chunk': args.chunk,
        'compiled_training_graph': args.compiled,
        'loss_scale': 1, 'nll_backward_reduction': 'sum', 'gradient_normalization': 'after-backward-fp32',
        'checkpoint': str(path), 'parent_manifest_sha256': sha256(path / 'mac_manifest.json'),
        'official_targets_advanced': 0, 'new_downloads': 0, 'production_promoted': False}
    try:
        with runtime.device_lock(Path(args.plan).parent):
            report({'phase': 'loading'})
            plan = json.loads(args.plan.read_text())
            state = runtime.read_checkpoint(path, sha256(args.plan))
            cache = runtime.stager_for(plan, 0)
            reader = cache.reader(state['data'])
            rows = reader.next_batch(1)
            result['rows_sha256'] = hashlib.sha256(rows.astype('<u2').tobytes()).hexdigest()
            result['measurements'] = (torch_case if args.backend == 'torch' else mlx_case)(args, path, state, rows, plan, report)
            last = result['measurements'][-1]
            result['valid_targets_per_second'] = last['valid_targets'] / last['seconds']
            result['completed'] = True
            write_json(root / 'result.json', result)
            report({'phase': 'completed', 'valid_targets_per_second': result['valid_targets_per_second']})
            print(json.dumps({k:v for k,v in result.items() if k != 'measurements'}), flush=True)
    except BaseException as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
        write_json(root / 'result.json', result)
        report({'phase': 'failed', 'error': result['error']})
        raise


if __name__ == '__main__':
    main()
