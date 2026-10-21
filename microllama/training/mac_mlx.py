"""Experimental MLX Engram/Qwen3 training graph for bounded speed comparisons.

No production launcher, dataset acquisition or checkpoint promotion. FP32
master parameters and explicit compute precision; the masked 8K objective,
shared memory insertion weights and tied output embedding are preserved.
"""
from __future__ import annotations
import math
from functools import lru_cache
from types import SimpleNamespace, MethodType

import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .engram import EngramMemory, EngramState, LOOKUP_V2


def scaled_value_and_grad(fun, *, compute='float32', compiled=False, loss_scale=None):
    """Keep dense vocabulary gradients representable during FP16 backprop."""
    scale = (8192. if compute == 'float16' else 1.) if loss_scale is None else float(loss_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite loss scale required')
    differentiated = mx.value_and_grad(lambda *args: fun(*args) * scale)
    if compiled:
        differentiated = mx.compile(differentiated)
    def evaluate(*args):
        value, gradients = differentiated(*args)
        return value / scale, tree_map(lambda g: g.astype(mx.float32) / scale, gradients)
    return evaluate


def memory_indices(config, ids, table_sizes):
    """Call the original signed-int64 implementation; no new hash arithmetic."""
    ids = torch.as_tensor(np.asarray(ids).astype(np.int64, copy=False))
    memory = SimpleNamespace(revised=True, heads=config['engram_hash_heads'],
        eos_token_id=config['eos_token_id'], pad_token_id=config['pad_token_id'],
        canonical_ids=torch.tensor(config['engram_token_id_map']),
        _BIGRAM_MULTIPLIERS=EngramMemory._BIGRAM_MULTIPLIERS,
        _TRIGRAM_MULTIPLIERS=EngramMemory._TRIGRAM_MULTIPLIERS,
        bigram_tables=[SimpleNamespace(num_embeddings=n) for n in table_sizes[:config['engram_hash_heads']]],
        trigram_tables=[SimpleNamespace(num_embeddings=n) for n in table_sizes[config['engram_hash_heads']:]])
    if config['engram_lookup_version'] != LOOKUP_V2:
        raise ValueError('This experimental path requires the original revised lookup')
    memory._is_boundary = MethodType(EngramMemory._is_boundary, memory)
    result = EngramMemory._indices(memory, ids, EngramState(torch.full((ids.shape[0], 2), -1)))
    return tuple(mx.array(x.numpy()) for x in result)


@mx.custom_function
def ordered_take(weight, ids):
    return weight[ids]


@ordered_take.vjp
def ordered_take_vjp(primals, cotangent, output):
    weight, ids = primals
    # CPU scatter order is explicit, matching the reproducible PyTorch adapter.
    with mx.stream(mx.cpu):
        gradient = mx.zeros(weight.shape, dtype=weight.dtype)
        gradient = gradient.at[ids.reshape(-1)].add(cotangent.reshape(-1, weight.shape[1]))
    return gradient, mx.zeros_like(ids)


def embedding(weight, ids, padding_idx=None):
    result = ordered_take(weight, ids)
    if padding_idx is not None:
        result = mx.where((ids != padding_idx)[..., None], result, mx.stop_gradient(result))
    return result


def rms(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps)


def qwen_rms(x, weight, eps):
    # Match Qwen3RMSNorm.forward: normalize in FP32, round to the
    # input dtype, then multiply by the FP32 learned weight. In mixed
    # precision q/k therefore return FP32 before RoPE, just as in CUDA.
    normalized = mx.fast.rms_norm(x.astype(mx.float32), None, eps)
    return normalized.astype(x.dtype) * weight.astype(mx.float32)


def attention_probabilities(q, k, first):
    """Flash-style FP32 score/softmax arithmetic on the original BF16 inputs."""
    batch, heads, length, dim = q.shape
    kv_heads = k.shape[1]
    grouped = q.astype(mx.float32).reshape(batch, kv_heads, heads // kv_heads, length, dim)
    keys = k.astype(mx.float32)[:, :, None]
    scores = (grouped * dim ** -.5) @ keys.swapaxes(-1, -2)
    allowed = mx.arange(k.shape[-2])[None, :] <= mx.arange(first, first + length)[:, None]
    return mx.softmax(mx.where(allowed, scores, -mx.inf), axis=-1, precise=True), grouped


@lru_cache(maxsize=16)
def attention_operator(tile):
    @mx.custom_function
    def operator(q, k, v):
        pieces = []
        step = tile or q.shape[-2]
        for first in range(0, q.shape[-2], step):
            last = min(first + step, q.shape[-2])
            query, keys, values = q[..., first:last, :], k[..., :last, :], v[..., :last, :]
            if mx.default_device() == mx.gpu and q.shape[-1] == 64 and last - first >= 16:
                part = mx.fast.scaled_dot_product_attention(query, keys, values,
                    scale=q.shape[-1] ** -.5, mask='causal', force_fused=True)
            else:
                probabilities, grouped = attention_probabilities(query, keys, first)
                part = (probabilities @ values.astype(mx.float32)[:, :, None]).reshape(query.shape).astype(q.dtype)
            pieces.append(part)
        return mx.concatenate(pieces, axis=-2)

    @operator.vjp
    def backward(primals, cotangent, output):
        q, k, v = primals
        step = tile or q.shape[-2]
        dq, dk, dv = [], mx.zeros(k.shape, mx.float32), mx.zeros(v.shape, mx.float32)
        batch, heads, length, dim = q.shape
        kv_heads, groups = k.shape[1], heads // k.shape[1]
        # MLX0.32's SDPA VJP falls back to BF16 score/probability matrices.
        # CUDA Flash instead keeps these intermediates and accumulated dK/dV
        # in FP32. Cast gradients back once after all query tiles/head groups.
        for first in range(0, length, step):
            last = min(first + step, length)
            query, keys, values = q[..., first:last, :], k[..., :last, :], v[..., :last, :]
            probabilities, grouped = attention_probabilities(query, keys, first)
            upstream = cotangent[..., first:last, :].astype(mx.float32).reshape(batch, kv_heads, groups, last-first, dim)
            dp = upstream @ values.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
            ds = probabilities * (dp - mx.sum(dp * probabilities, axis=-1, keepdims=True))
            dq.append(((ds @ keys.astype(mx.float32)[:, :, None]) * dim ** -.5).reshape(query.shape))
            key_gradient = mx.sum(ds.swapaxes(-1, -2) @ grouped, axis=2) * dim ** -.5
            value_gradient = mx.sum(probabilities.swapaxes(-1, -2) @ upstream, axis=2)
            dk = dk.at[..., :last, :].add(key_gradient)
            dv = dv.at[..., :last, :].add(value_gradient)
        return mx.concatenate(dq, axis=-2).astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return operator


def attention(q, k, v, tile):
    return attention_operator(tile)(q, k, v)


def silu(x):
    # Match PyTorch/CUDA's fused SiLU opmath, without rounding the sigmoid
    # intermediate to BF16. The activation output/gradient retain input dtype.
    return nn.silu(x.astype(mx.float32)).astype(x.dtype)


def hidden_state(params, config, ids, indices, *, compute='float32', query_tile=512, checkpoint=True):
    if compute not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Explicit supported compute precision required')
    dtype = {'float32': mx.float32, 'float16': mx.float16, 'bfloat16': mx.bfloat16}[compute]
    eps, heads, kv_heads, dim = (config[k] for k in ('rms_norm_eps', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))
    x = embedding(params['model.embed_tokens.weight'], ids, config['pad_token_id'])
    def linear(x, weight):
        return x.astype(dtype) @ weight.astype(dtype).T
    def block(weights, x):
        normalized = qwen_rms(x, weights['input_layernorm.weight'], eps)
        shape = (*x.shape[:2], -1, dim)
        q = linear(normalized, weights['self_attn.q_proj.weight']).reshape(shape)
        k = linear(normalized, weights['self_attn.k_proj.weight']).reshape(shape)
        v = linear(normalized, weights['self_attn.v_proj.weight']).reshape(shape).transpose(0, 2, 1, 3)
        q = qwen_rms(q, weights['self_attn.q_norm.weight'], eps).transpose(0, 2, 1, 3)
        k = qwen_rms(k, weights['self_attn.k_norm.weight'], eps).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, dims=dim, traditional=False, base=config['rope_theta'], scale=1., offset=0)
        k = mx.fast.rope(k, dims=dim, traditional=False, base=config['rope_theta'], scale=1., offset=0)
        # Autocast on NVIDIA casts q/k/v at the SDPA boundary, after RoPE.
        attended = attention(q.astype(dtype), k.astype(dtype), v, query_tile).transpose(0, 2, 1, 3).reshape(*x.shape[:2], heads * dim)
        x = x + linear(attended, weights['self_attn.o_proj.weight']).astype(mx.float32)
        normalized = qwen_rms(x, weights['post_attention_layernorm.weight'], eps)
        gate = silu(linear(normalized, weights['mlp.gate_proj.weight']))
        return x + linear(gate * linear(normalized, weights['mlp.up_proj.weight']),
                          weights['mlp.down_proj.weight']).astype(mx.float32)
    bigram, trigram, bigram_valid, trigram_valid = indices
    def memory(weights, x):
        b = mx.concatenate([embedding(weights[f'bigram_tables.{h}.weight'], bigram[..., h])
                            for h in range(config['engram_hash_heads'])], axis=-1)
        t = mx.concatenate([embedding(weights[f'trigram_tables.{h}.weight'], trigram[..., h])
                            for h in range(config['engram_hash_heads'])], axis=-1)
        values = mx.concatenate([b * bigram_valid[..., None], t * trigram_valid[..., None]], axis=-1)
        key_input = linear(values, weights['key_proj.weight'])
        # Engram uses torch.nn.RMSNorm, whose mixed-precision output
        # rounds after the weight multiplication and retains input dtype.
        key = rms(key_input, weights['key_norm.weight'], eps).astype(key_input.dtype)
        query = rms(x, weights['query_norm.weight'], eps)
        gate = mx.sigmoid(mx.sum(query * key, axis=-1, keepdims=True) / math.sqrt(config['hidden_size']))
        valid = (ids != config['eos_token_id']) & (ids != config['pad_token_id'])
        return x + gate * linear(values, weights['value_proj.weight']).astype(mx.float32) * valid[..., None] * weights['residual_scale']
    mem_weights = {k.removeprefix('model.engram.'): v for k, v in params.items() if k.startswith('model.engram.')}
    for index in range(config['num_hidden_layers']):
        prefix = f'model.layers.{index}.'
        weights = {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}
        x = mx.checkpoint(block)(weights, x) if checkpoint else block(weights, x)
        if index in config['engram_layers']:
            x = mx.checkpoint(memory)(mem_weights, x) if checkpoint else memory(mem_weights, x)
    return qwen_rms(x, params['model.norm.weight'], eps)


def loss(params, config, rows, indices, *, compute='float32', query_tile=512, chunk=256, checkpoint=True,
         reduction='mean'):
    if reduction not in ('sum', 'mean'):
        raise ValueError('Explicit summed or mean objective required')
    ids, labels = rows[:, :-1], rows[:, 1:]
    valid = (ids != config['eos_token_id']) & (ids != config['pad_token_id']) & (labels != config['pad_token_id'])
    hidden = hidden_state(params, config, ids, indices, compute=compute, query_tile=query_tile, checkpoint=checkpoint)
    hidden, labels, valid = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1), valid.reshape(-1)
    dtype = {'float32': mx.float32, 'float16': mx.float16, 'bfloat16': mx.bfloat16}[compute]
    def part(h, w, y, mask):
        y, mask = mx.stop_gradient(y), mx.stop_gradient(mask)
        logits = (h.astype(dtype) @ w.astype(dtype).T).astype(mx.float32)
        return mx.sum(nn.losses.cross_entropy(logits, y, reduction='none') * mask)
    total = mx.array(0., mx.float32)
    for start in range(0, labels.size, chunk):
        args = (hidden[start:start + chunk], params['model.embed_tokens.weight'],
                labels[start:start + chunk], valid[start:start + chunk])
        total = total + (mx.checkpoint(part)(*args) if checkpoint else part(*args))
    return total if reduction == 'sum' else total / mx.sum(valid)


def parameters_from_torch(model):
    return {name: mx.array(value.detach().cpu().float().numpy()) for name, value in model.named_parameters()}


def adam_from_torch(model, optimizer, saved):
    names = {id(value): name for name, value in model.named_parameters()}
    slots, policy = {}, {}
    for live, stored in zip(optimizer.param_groups, saved['param_groups'], strict=True):
        if (len(live['params']) != len(stored['params']) or live.get('group_name') != stored.get('group_name')
                or stored.get('amsgrad', False) or stored.get('maximize', False)):
            raise ValueError('Adam group order differs')
        for parameter, identifier in zip(live['params'], stored['params'], strict=True):
            name, state = names[id(parameter)], saved['state'][identifier]
            slots[name] = {'m': mx.array(state['exp_avg'].cpu().float().numpy()),
                           'v': mx.array(state['exp_avg_sq'].cpu().float().numpy()),
                           'step': int(state['step'])}
            policy[name] = {key: stored[key] for key in ('lr', 'weight_decay', 'betas', 'eps')}
    if set(slots) != set(names.values()):
        raise ValueError('Every trainable parameter requires original Adam state')
    return slots, policy


def adam_update(params, gradients, slots, policy, clip=1.):
    # Canonical reduction order survives a fresh process/Python hash seed.
    norm = mx.sqrt(sum(mx.sum(gradients[name].astype(mx.float32) ** 2) for name in sorted(gradients)))
    scale = mx.minimum(mx.array(1., mx.float32), clip / (norm + 1e-6))
    updated, next_slots = {}, {}
    for name in sorted(params):
        weight = params[name]
        g, state, settings = gradients[name] * scale, slots[name], policy[name]
        beta1, beta2 = settings['betas']
        m = beta1 * state['m'] + (1 - beta1) * g
        v = beta2 * state['v'] + (1 - beta2) * g * g
        step = state['step'] + 1
        denom = mx.sqrt(v) / math.sqrt(1 - beta2 ** step) + settings['eps']
        updated[name] = weight * (1 - settings['lr'] * settings['weight_decay']) - (
            settings['lr'] / (1 - beta1 ** step)) * m / denom
        next_slots[name] = {'m': m, 'v': v, 'step': step}
    return updated, next_slots, norm
