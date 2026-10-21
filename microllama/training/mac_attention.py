"""Full-context causal SDPA with bounded query tiles and recomputed backward.

No token/context truncation or approximate attention. The largest attention
matrix has query_tile rows instead of the full sequence length.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NAME = 'mac_tiled_sdpa'


def tiled_sdpa(query, key, value, mask=None, *, query_tile=256, causal=True, scaling=None):
    if type(query_tile) is not int or query_tile < 1:
        raise ValueError('Positive query tile required')
    length, keys = query.shape[-2], key.shape[-2]
    if causal and length != keys:
        raise ValueError('Tiled causal training requires equal query/key sequence lengths')
    pieces = []
    for first in range(0, length, query_tile):
        last = min(length, first + query_tile)
        end = last if causal else keys
        q, k, v = query[..., first:last, :], key[..., :end, :], value[..., :end, :]
        if mask is None:
            tile_mask = (torch.arange(end, device=query.device)[None, :] <=
                         torch.arange(first, last, device=query.device)[:, None]) if causal else None
        else:
            tile_mask = mask[..., first:last, :end] if mask.shape[-2] != 1 else mask[..., :end]
            if causal:
                allowed = (torch.arange(end, device=query.device)[None, :] <=
                           torch.arange(first, last, device=query.device)[:, None])
                tile_mask = tile_mask & allowed if tile_mask.dtype == torch.bool else tile_mask.masked_fill(~allowed, -torch.inf)
        def attention(a, b, c, m):
            return F.scaled_dot_product_attention(a, b, c, attn_mask=m, dropout_p=0.,
                                                  is_causal=False, scale=scaling)
        if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
            output = checkpoint(attention, q, k, v, tile_mask, use_reentrant=False, preserve_rng_state=True)
        else:
            output = attention(q, k, v, tile_mask)
        pieces.append(output)
    return torch.cat(pieces, dim=-2)


def attention_forward(module, query, key, value, attention_mask, dropout=0., scaling=None,
                      is_causal=None, **kwargs):
    from transformers.integrations.sdpa_attention import repeat_kv
    if dropout != 0 or kwargs.get('output_attentions') or kwargs.get('head_mask') is not None or kwargs.get('sliding_window') is not None:
        raise ValueError('Mac attention supports zero-dropout full causal attention only')
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    causal = (query.shape[-2] > 1 and attention_mask is None and getattr(module, 'is_causal', True)
              if is_causal is None else bool(is_causal))
    if attention_mask is not None:
        attention_mask = attention_mask[..., :key.shape[-2]]
    output = tiled_sdpa(query, key, value, attention_mask,
        query_tile=module._mac_query_tile, causal=causal, scaling=scaling)
    return output.transpose(1, 2).contiguous(), None


def install_tiled_attention(model, query_tile=256):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    if model.config.attention_dropout != 0 or any(
        layer.self_attn.sliding_window is not None for layer in model.model.layers):
        raise ValueError('Tiled attention requires the unchanged full-attention/no-dropout model')
    ALL_ATTENTION_FUNCTIONS.register(NAME, attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
    for layer in model.model.layers:
        layer.self_attn._mac_query_tile = query_tile
    model.set_attn_implementation(NAME)
