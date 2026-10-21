"""Ordered GPU reduction must exactly match CPU scatter, including PAD/head."""
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core',exc_type=ImportError)
from microllama.training.mac_gpu_embedding import scatter,embedding


@pytest.fixture(autouse=True)
def restore_device():
    previous=mx.default_device();mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('length,rows,dim',[(8192,128,64),(8192,65536,48),(257,128,1024)])
def test_stable_gpu_reduction_matches_original_order_and_repeats_exactly(length,rows,dim):
    rng=np.random.default_rng(528)
    ids=rng.integers(0,min(rows,100),size=(1,length),dtype=np.int32)
    values=rng.normal(size=(1,length,dim)).astype(np.float32)
    # Large/small alternating terms make reassociation/order changes visible.
    values[:,::3]*=1000
    expected=np.zeros((rows,dim),np.float32)
    np.add.at(expected,ids.reshape(-1),values.reshape(-1,dim))
    with mx.stream(mx.cpu):
        cpu=mx.zeros((rows,dim),mx.float32).at[mx.array(ids).reshape(-1)].add(mx.array(values).reshape(-1,dim))
    np.testing.assert_array_equal(np.asarray(cpu),expected)
    for _ in range(3):
        actual=scatter(mx.array(ids),mx.array(values),(rows,dim));mx.eval(actual)
        np.testing.assert_array_equal(np.asarray(actual),expected)


def test_gpu_embedding_preserves_pad_forward_and_tied_head_contribution():
    rng=np.random.default_rng(152)
    weights=rng.normal(size=(32,64)).astype(np.float32)
    ids=np.array([[0,2,2,0,3,3,3,4]],np.int32)
    upstream=rng.normal(size=(1,8,64)).astype(np.float32)
    mask=ids!=0
    expected=np.zeros_like(weights)
    np.add.at(expected,ids[mask],upstream[mask])
    expected+=2*weights
    fn=mx.compile(mx.value_and_grad(lambda w: mx.sum(embedding(w,mx.array(ids),0)*mx.array(upstream))+mx.sum(w*w)))
    _,gradient=fn(mx.array(weights));mx.eval(gradient)
    np.testing.assert_array_equal(np.asarray(gradient),expected)
    np.testing.assert_array_equal(np.asarray(embedding(mx.array(weights),mx.array(ids),0)),weights[ids])
