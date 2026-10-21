"""Deterministic FP32 embedding gradients with stable-sort GPU row reduction.

Each unique ID has one threadgroup; each component sums original token positions
in order. No floating-point atomics or parallel reassociation are used.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(name='microllama_ordered_embedding_scatter',
        input_names=['ids','positions','cotangent'], output_names=['gradient'],
        source=Path(__file__).with_name('metal').joinpath('embedding_scatter.metal').read_text())


def scatter(ids, cotangent, shape):
    if cotangent.dtype != mx.float32:
        raise ValueError('Deterministic GPU embedding reduction requires FP32 gradients')
    flat=ids.reshape(-1)
    if flat.size == 0:
        return mx.zeros(shape,mx.float32)
    # MLX0.32.4 argsort is stable: equal IDs retain original token order.
    positions=mx.argsort(flat)
    return _kernel()(inputs=[flat,positions,cotangent.reshape(-1,shape[1])],
        template=[('L',flat.size),('D',shape[1])],grid=(flat.size*256,1,1),
        threadgroup=(256,1,1),output_shapes=[shape],output_dtypes=[mx.float32],
        init_value=0)[0]


@mx.custom_function
def ordered_take(weight,ids):
    return weight[ids]


@ordered_take.vjp
def _vjp(primals,cotangent,output):
    weight,ids=primals
    if mx.default_device() == mx.gpu and weight.dtype == mx.float32:
        gradient=scatter(ids,cotangent.astype(mx.float32),weight.shape)
    else:
        with mx.stream(mx.cpu):
            gradient=mx.zeros(weight.shape,weight.dtype).at[ids.reshape(-1)].add(cotangent.reshape(-1,weight.shape[1]))
    return gradient,mx.zeros_like(ids)


def embedding(weight,ids,padding_idx=None):
    result=ordered_take(weight,ids)
    if padding_idx is not None:
        result=mx.where((ids!=padding_idx)[...,None],result,mx.stop_gradient(result))
    return result
