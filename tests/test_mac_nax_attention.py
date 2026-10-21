"""Independent causal attention derivatives and deterministic custom GPU writes."""
import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training.mac_nax_attention import attention, supported


@pytest.fixture(autouse=True)
def gpu():
    previous=mx.default_device(); mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('length', [16, 17, 33, 64, 129])
def test_custom_attention_matches_independent_fp32_cpu_derivatives(length):
    rng=np.random.default_rng(861+length)
    arrays=[mx.array(rng.normal(size=(2,2,length,64)).astype(np.float32)).astype(mx.bfloat16) for _ in range(4)]
    q,k,v,g=arrays
    inputs=[torch.tensor(np.asarray(a.astype(mx.float32)),requires_grad=True) for a in arrays[:3]]
    scores=(inputs[0]@inputs[1].transpose(-1,-2))*.125
    mask=torch.arange(length)[None,:]<=torch.arange(length)[:,None]
    reference=torch.softmax(scores.masked_fill(~mask,-torch.inf),dim=-1)@inputs[2]
    reference.backward(torch.tensor(np.asarray(g.astype(mx.float32))))
    output, gradients=mx.vjp(attention,[q,k,v],[g]); mx.eval(output,gradients)
    native=mx.fast.scaled_dot_product_attention(q,k,v,scale=.125,mask='causal')
    np.testing.assert_array_equal(np.asarray(output[0].astype(mx.float32)),np.asarray(native.astype(mx.float32)))
    for actual,expected in zip(gradients,inputs):
        a=np.asarray(actual.astype(mx.float32)).reshape(-1)
        b=expected.grad.numpy().reshape(-1)
        assert np.isfinite(a).all()
        if np.linalg.norm(b)==0:
            np.testing.assert_allclose(a,b,atol=.01)
        else:
            assert np.linalg.norm(a-b)/np.linalg.norm(b)<.02
            assert np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))>.999


def test_custom_backward_is_bit_exact_for_repeated_calls_and_noncontiguous_inputs():
    mx.random.seed(823)
    inputs=[mx.random.normal((2,65,2,64)).astype(mx.bfloat16).transpose(0,2,1,3) for _ in range(3)]
    cotangent=mx.random.normal(inputs[0].shape).astype(mx.bfloat16)
    fn=mx.compile(lambda q,k,v,g: mx.vjp(attention,[q,k,v],[g]))
    repeats=[]
    for _ in range(3):
        _,gradients=fn(*inputs,cotangent); mx.eval(gradients)
        repeats.append([np.asarray(g.astype(mx.float32)) for g in gradients])
    for repeated in repeats[1:]:
        for a,b in zip(repeats[0],repeated): np.testing.assert_array_equal(a,b)


def test_grouped_query_and_fp32_inputs_keep_native_path():
    q=mx.ones((1,2,17,64),mx.float32)
    k=v=mx.ones((1,1,17,64),mx.float32)
    assert not supported(q,k,v)
    expected=mx.fast.scaled_dot_product_attention(q,k,v,scale=.125,mask='causal')
    np.testing.assert_array_equal(np.asarray(attention(q,k,v)),np.asarray(expected))
