"""Native MLX Qwen3/Engram graph for large-memory Apple silicon.

Uses the framework's SDPA forward/backward and fused cross entropy. The model,
document mask, signed-int64 Engram hashes, tied/PAD gradients and FP32 optimizer
state are unchanged. The original M1 implementation remains available separately.
This module contains no launcher and never starts production automatically.
"""
from __future__ import annotations

import math
import time

import mlx.core as mx

from .mac_mlx import embedding, memory_indices, qwen_rms, rms, silu, scaled_value_and_grad
from .mac_ultra_run import BatchEngine


def hidden_state(params, config, ids, indices, *, compute='bfloat16', checkpoint=False, fused_projections=False,
                 gpu_embedding=False):
    if compute not in ('float32', 'bfloat16'):
        raise ValueError('FP32 diagnostic or BF16 mixed-precision compute required')
    dtype = mx.float32 if compute == 'float32' else mx.bfloat16
    eps = config['rms_norm_eps']
    heads, dim = config['num_attention_heads'], config['head_dim']
    lookup = embedding
    if gpu_embedding:
        from .mac_gpu_embedding import embedding as lookup
    x = lookup(params['model.embed_tokens.weight'], ids, config['pad_token_id'])
    def linear(x, weight):
        return x.astype(dtype) @ weight.astype(dtype).T
    def block(weights, x):
        norm = qwen_rms(x, weights['input_layernorm.weight'], eps)
        shape = (*x.shape[:2], -1, dim)
        if fused_projections:
            names = [f'self_attn.{n}_proj.weight' for n in ('q','k','v')]
            sizes = [weights[n].shape[0] for n in names]
            joined = mx.concatenate([weights[n].astype(dtype) for n in names], axis=0)
            q,k,v = mx.split(norm.astype(dtype)@joined.T, [sizes[0],sizes[0]+sizes[1]], axis=-1)
            q,k,v = q.reshape(shape),k.reshape(shape),v.reshape(shape).transpose(0,2,1,3)
        else:
            q = linear(norm, weights['self_attn.q_proj.weight']).reshape(shape)
            k = linear(norm, weights['self_attn.k_proj.weight']).reshape(shape)
            v = linear(norm, weights['self_attn.v_proj.weight']).reshape(shape).transpose(0, 2, 1, 3)
        q = qwen_rms(q, weights['self_attn.q_norm.weight'], eps).transpose(0, 2, 1, 3)
        k = qwen_rms(k, weights['self_attn.k_norm.weight'], eps).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, dims=dim, traditional=False, base=config['rope_theta'], scale=1., offset=0)
        k = mx.fast.rope(k, dims=dim, traditional=False, base=config['rope_theta'], scale=1., offset=0)
        attended = mx.fast.scaled_dot_product_attention(q.astype(dtype), k.astype(dtype), v,
                                                       scale=dim**-.5, mask='causal')
        attended = attended.transpose(0, 2, 1, 3).reshape(*x.shape[:2], heads*dim)
        x = x + linear(attended, weights['self_attn.o_proj.weight']).astype(mx.float32)
        norm = qwen_rms(x, weights['post_attention_layernorm.weight'], eps)
        if fused_projections:
            joined = mx.concatenate([weights[f'mlp.{n}_proj.weight'].astype(dtype) for n in ('gate','up')], axis=0)
            gate,up = mx.split(norm.astype(dtype)@joined.T, 2, axis=-1)
            gate = silu(gate)
        else:
            gate = silu(linear(norm, weights['mlp.gate_proj.weight']))
            up = linear(norm, weights['mlp.up_proj.weight'])
        return x + linear(gate*up,
                          weights['mlp.down_proj.weight']).astype(mx.float32)
    bigram, trigram, bigram_valid, trigram_valid = indices
    def memory(weights, x):
        b = mx.concatenate([lookup(weights[f'bigram_tables.{h}.weight'], bigram[..., h])
                            for h in range(config['engram_hash_heads'])], axis=-1)
        t = mx.concatenate([lookup(weights[f'trigram_tables.{h}.weight'], trigram[..., h])
                            for h in range(config['engram_hash_heads'])], axis=-1)
        values = mx.concatenate([b*bigram_valid[..., None], t*trigram_valid[..., None]], axis=-1)
        key_input = linear(values, weights['key_proj.weight'])
        key = rms(key_input, weights['key_norm.weight'], eps).astype(key_input.dtype)
        query = rms(x, weights['query_norm.weight'], eps)
        gate = mx.sigmoid(mx.sum(query*key, axis=-1, keepdims=True)/math.sqrt(config['hidden_size']))
        valid = (ids != config['eos_token_id']) & (ids != config['pad_token_id'])
        return x + gate*linear(values, weights['value_proj.weight']).astype(mx.float32)*valid[..., None]*weights['residual_scale']
    memory_weights = {n.removeprefix('model.engram.'): v for n, v in params.items() if n.startswith('model.engram.')}
    for index in range(config['num_hidden_layers']):
        prefix = f'model.layers.{index}.'
        weights = {n[len(prefix):]: v for n, v in params.items() if n.startswith(prefix)}
        x = mx.checkpoint(block)(weights, x) if checkpoint else block(weights, x)
        if index in config['engram_layers']:
            x = mx.checkpoint(memory)(memory_weights, x) if checkpoint else memory(memory_weights, x)
    return qwen_rms(x, params['model.norm.weight'], eps)


def loss(params, config, rows, indices, *, compute='bfloat16', chunk=1024, checkpoint=False, reduction='sum', fused_projections=False,
         gpu_embedding=False):
    if reduction not in ('sum', 'mean') or type(chunk) is not int or chunk < 1:
        raise ValueError('Positive loss chunk and explicit sum/mean reduction required')
    ids, labels = rows[:, :-1], rows[:, 1:]
    valid = (ids != config['eos_token_id']) & (ids != config['pad_token_id']) & (labels != config['pad_token_id'])
    h = hidden_state(params, config, ids, indices, compute=compute, checkpoint=checkpoint,
                     fused_projections=fused_projections,gpu_embedding=gpu_embedding)
    h, labels, valid = h.reshape(-1, h.shape[-1]), labels.reshape(-1), valid.reshape(-1)
    dtype = mx.float32 if compute == 'float32' else mx.bfloat16
    weight = params['model.embed_tokens.weight']
    total = mx.array(0., mx.float32)
    def part(hidden, weight, targets, mask):
        logits = (hidden.astype(dtype) @ weight.astype(dtype).T).astype(mx.float32)
        return mx.sum(mx.fast.cross_entropy(logits, mx.stop_gradient(targets))*mx.stop_gradient(mask))
    for first in range(0, labels.size, chunk):
        args = (h[first:first+chunk], weight, labels[first:first+chunk], valid[first:first+chunk])
        total = total + (mx.checkpoint(part)(*args) if checkpoint else part(*args))
    return total if reduction == 'sum' else total/mx.sum(valid)


class FastEngine(BatchEngine):
    """FP32 masters/Adam and normalized global16 updates through native MLX ops."""
    def __init__(self, params, slots, policy, config, *, micro_batch_size=1,
                 chunk=1024, checkpoint=False, accumulation_window=1, fused_projections=False,
                 gpu_embedding=False, extras=None):
        if type(accumulation_window) is not int or accumulation_window not in (1, 2):
            raise ValueError('Only bounded one/two-microbatch evaluation is supported')
        super().__init__(params, slots, policy, config, micro_batch_size=micro_batch_size,
                         query_tile=512, chunk=chunk, extras=extras)
        self.checkpoint = checkpoint
        self.accumulation_window = accumulation_window
        self.fused_projections = fused_projections
        self.gpu_embedding = gpu_embedding
        self.gradient = scaled_value_and_grad(lambda p, rows, indices: loss(p, config, rows, indices,
            compute='bfloat16', chunk=chunk, checkpoint=checkpoint, reduction='sum', fused_projections=fused_projections,
            gpu_embedding=gpu_embedding),
            compute='bfloat16', compiled=True, loss_scale=1)
        self.forward = mx.compile(lambda p, rows, indices: loss(p, config, rows, indices,
            compute='bfloat16', chunk=chunk, checkpoint=False, reduction='sum', fused_projections=fused_projections,
            gpu_embedding=gpu_embedding))

    def gradients(self, rows, report=lambda value: None):
        if self.accumulation_window == 1:
            return super().gradients(rows, report)
        if rows.ndim != 2 or rows.shape[0] != 16 or rows.shape[1] < 2:
            raise ValueError('A complete ordered global16 batch is required')
        total, summed, accumulated = 0, mx.array(0., mx.float32), None
        started = time.monotonic()
        for first in range(0, 16, self.micro_batch_size):
            batch = rows[first:first+self.micro_batch_size]
            valid = int(((batch[:, :-1] != 7) & (batch[:, :-1] != 0) & (batch[:, 1:] != 0)).sum())
            if valid < 1:
                raise ValueError('No valid targets in microbatch')
            value, gradients = self.gradient(self.params, mx.array(batch), self.indices(batch))
            accumulated = {n: gradients[n] + (0 if accumulated is None else accumulated[n])
                           for n in sorted(gradients)}
            summed = summed + value
            total += valid
            done = first//self.micro_batch_size+1
            mx.async_eval(summed, accumulated)
            # Bound outstanding graphs/activation memory; read scalars only here.
            if done % self.accumulation_window == 0 or first+self.micro_batch_size == 16:
                mx.eval(summed, accumulated)
                if not math.isfinite(float(summed)):
                    raise RuntimeError('Nonfinite summed training loss')
                report({'microbatches_done': done, 'microbatches_total': 16//self.micro_batch_size,
                        'sequences_done': first+self.micro_batch_size,
                        'physical_batch_size': self.micro_batch_size,
                        'valid_targets_so_far': total, 'seconds': time.monotonic()-started})
        return float(summed)/total, {n: accumulated[n]/total for n in sorted(accumulated)}, total
