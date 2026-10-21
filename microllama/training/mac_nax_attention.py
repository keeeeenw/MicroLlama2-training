"""Experimental causal D64 MHA backward using Metal4 cooperative tensors.

The forward is native MLX. This opt-in operator is not a production default.
Every output gradient has a unique writer and a fixed accumulation order.
Unsupported shapes/dtypes/devices use native MLX, never an approximate fallback.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def kernels():
    folder = Path(__file__).with_name('metal')
    header = (folder/'nax_attention.h').read_text()
    return tuple(mx.fast.metal_kernel(name='microllama_nax_'+name,
        input_names=inputs, output_names=outputs, header=header,
        source=(folder/(name+'.metal')).read_text())
        for name, inputs, outputs in [
            ('attention_stats', ['q','k','o','g'], ['stats']),
            ('attention_dq', ['q','k','v','g','stats'], ['dq']),
            ('attention_dkv_shared', ['q','k','v','g','stats'], ['dk','dv'])])


def backward(q, k, v, g, o):
    if not supported(q,k,v):
        raise ValueError('Custom backward requires GPU BF16 same-shape D64 MHA')
    length = q.shape[2]
    count = q.shape[0]*q.shape[1]
    g = g.astype(q.dtype)
    stats_kernel, dq_kernel, dkv_kernel = kernels()
    kwargs = dict(template=[('T', mx.bfloat16), ('N', length)],
                  grid=((length+15)//16*count*32,1,1), threadgroup=(32,1,1))
    stats = stats_kernel(inputs=[q,k,o,g], output_shapes=[(*q.shape[:3],2)],
                         output_dtypes=[mx.float32], **kwargs)[0]
    dq = dq_kernel(inputs=[q,k,v,g,stats], output_shapes=[q.shape],
                   output_dtypes=[q.dtype], **kwargs)[0]
    dkv_kwargs={**kwargs,'grid':((length+63)//64*count*128,1,1),'threadgroup':(128,1,1)}
    dk,dv = dkv_kernel(inputs=[q,k,v,g,stats], output_shapes=[k.shape,v.shape],
                       output_dtypes=[k.dtype,v.dtype], **dkv_kwargs)
    return dq,dk,dv


@lru_cache(maxsize=1)
def _nax_device():
    # This experimental specialization is qualified only on the M5 GPU family.
    return 'Apple M5' in mx.device_info(mx.gpu)['device_name']


def supported(q,k,v):
    return (mx.default_device() == mx.gpu and q.ndim == 4 and q.shape == k.shape == v.shape
            and q.shape[2] >= 16 and q.shape[-1] == 64
            and q.dtype == k.dtype == v.dtype == mx.bfloat16 and _nax_device())


@mx.custom_function
def _operator(q,k,v):
    return mx.fast.scaled_dot_product_attention(q,k,v,scale=.125,mask='causal')


@_operator.vjp
def _vjp(primals, cotangent, output):
    return backward(*primals,cotangent,output)


def attention(q,k,v):
    if supported(q,k,v):
        return _operator(q,k,v)
    return mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-.5,mask='causal')
