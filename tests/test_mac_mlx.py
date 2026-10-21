"""CPU graph parity for the experimental MLX speed path; no production run."""
import copy
import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training.mac_mlx import loss, memory_indices, parameters_from_torch, adam_from_torch, adam_update, ordered_take, scaled_value_and_grad
from microllama.training.mac_mlx import silu,attention
from microllama.training.objective import chunked_loss, target_mask
from test_mac_training import model
from microllama.training.mac import update
from microllama.training.optimizers import build_optimizer


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('tile', [0, 5])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_mlx_engram_masked_forward_and_all_parameter_gradients(checkpoint, tile, compiled, device):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    try:
        torch.manual_seed(571)
        reference = model()
        rows = torch.tensor([[0, 2, 3, 13, 7, 12, 13, 0, 15, 16, 17, 18],
                             [3, 3, 3, 3, 7, 7, 5, 6, 0, 0, 10, 11]])
        x, y = rows[:, :-1], rows[:, 1:]
        expected = chunked_loss(reference, x, y, chunk_size=4, loss_mask=target_mask(x, y, 7, 0))
        expected.backward()
        params, config = parameters_from_torch(reference), reference.config.to_dict()
        sizes = [reference.model.engram.bigram_tables[h].num_embeddings for h in range(4)]
        sizes += [reference.model.engram.trigram_tables[h].num_embeddings for h in range(4)]
        indices = memory_indices(config, x.numpy(), sizes)
        fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices,
            query_tile=tile, checkpoint=checkpoint, chunk=4), compiled=compiled)
        value, gradients = fn(params, mx.array(rows.numpy()), indices)
        mx.eval(value, gradients)
        assert float(value) == pytest.approx(float(expected.detach()), abs=2e-6)
        for name, value in reference.named_parameters():
            np.testing.assert_allclose(np.array(gradients[name]), value.grad.numpy(), atol=2e-6, rtol=3e-4, err_msg=name)
        assert np.abs(np.array(gradients['model.embed_tokens.weight'])[0]).sum() > 0
    finally:
        mx.set_default_device(previous)


def test_mlx_preserves_imported_adam_moments_groups_and_next_step():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        torch.manual_seed(52)
        reference = model()
        optimizer = build_optimizer(reference, fused=False, memory_lr_multiplier=5)
        rows = torch.tensor([[2, 3, 4, 7, 8, 9, 0, 10]])
        update(reference, optimizer, [(rows[:, :-1], rows[:, 1:])], chunk_size=4)
        params = parameters_from_torch(reference)
        slots, policy = adam_from_torch(reference, optimizer, copy.deepcopy(optimizer.state_dict()))
        gradients = {}
        for name, parameter in reference.named_parameters():
            parameter.grad = torch.randn_like(parameter) * .01
            gradients[name] = mx.array(parameter.grad.numpy())
        result, moments, _ = adam_update(params, gradients, slots, policy)
        mx.eval(result, moments)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        optimizer.step()
        for name, parameter in reference.named_parameters():
            np.testing.assert_allclose(np.array(result[name]), parameter.detach().numpy(), atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(np.array(moments[name]['m']), optimizer.state[parameter]['exp_avg'].numpy(), atol=1e-8, rtol=1e-6)
            np.testing.assert_allclose(np.array(moments[name]['v']), optimizer.state[parameter]['exp_avg_sq'].numpy(), atol=1e-9, rtol=1e-6)
            assert moments[name]['step'] == int(optimizer.state[parameter]['step'])
    finally:
        mx.set_default_device(previous)


def test_mlx_cpu_embedding_reduction_is_repeatable_for_duplicate_ids():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        generator = np.random.default_rng(189)
        weight = mx.array(generator.normal(size=(128, 64)).astype(np.float32))
        ids = mx.array(generator.integers(0, 16, size=(1, 8192), dtype=np.int32))
        upstream = mx.array(generator.normal(size=(1, 8192, 64)).astype(np.float32))
        fn = mx.grad(lambda w: mx.sum(ordered_take(w, ids) * upstream))
        values = [np.array(fn(weight)) for _ in range(4)]
        for actual in values[1:]:
            np.testing.assert_array_equal(actual, values[0])
    finally:
        mx.set_default_device(previous)


def test_mlx_token_weighted_micro_accumulation_matches_batched_gradients():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        torch.manual_seed(92)
        reference = model()
        rows = np.array([[2, 3, 7, 8, 9, 0, 0, 0], [2, 3, 4, 5, 6, 8, 9, 10]], dtype=np.int32)
        params, config = parameters_from_torch(reference), reference.config.to_dict()
        sizes = [table.num_embeddings for table in reference.model.engram.bigram_tables]
        sizes += [table.num_embeddings for table in reference.model.engram.trigram_tables]
        counts = ((rows[:, :-1] != 7) & (rows[:, :-1] != 0) & (rows[:, 1:] != 0)).sum(axis=1)
        assert counts[0] != counts[1]
        fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices,
            query_tile=5, checkpoint=True, chunk=4), compiled=True)
        expected_loss, expected = fn(params, mx.array(rows), memory_indices(config, rows[:, :-1], sizes))
        accumulated, means = None, []
        for index, row in enumerate(rows):
            value, gradients = fn(params, mx.array(row[None]), memory_indices(config, row[None, :-1], sizes))
            weight = float(counts[index]) / int(counts.sum())
            accumulated = {name: g * weight + (0 if accumulated is None else accumulated[name])
                           for name, g in gradients.items()}
            means.append(float(value))
            mx.eval(accumulated)
        assert np.dot(means, counts) / counts.sum() == pytest.approx(float(expected_loss), abs=2e-6)
        for name in expected:
            np.testing.assert_allclose(np.array(accumulated[name]), np.array(expected[name]), atol=2e-6, rtol=3e-4)
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_mlx_bf16_compute_keeps_fp32_masters_and_gradients(device):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    try:
        torch.manual_seed(619)
        reference = model()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if 'norm' in name and name.endswith('weight'):
                    parameter.copy_(1 + torch.randn_like(parameter) * .1)
        rows = torch.tensor([[2, 3, 3, 7, 8, 9, 0, 0], [2, 3, 4, 5, 6, 8, 9, 10]])
        x, y = rows[:, :-1], rows[:, 1:]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected = chunked_loss(reference, x, y, chunk_size=4, loss_mask=target_mask(x, y, 7, 0))
        expected.backward()
        params, config = parameters_from_torch(reference), reference.config.to_dict()
        sizes = [table.num_embeddings for table in reference.model.engram.bigram_tables]
        sizes += [table.num_embeddings for table in reference.model.engram.trigram_tables]
        fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices,
            compute='bfloat16', query_tile=5, checkpoint=True, chunk=4), compute='bfloat16', compiled=True)
        value, gradients = fn(params, mx.array(rows.numpy()), memory_indices(config, x.numpy(), sizes))
        mx.eval(value, gradients)
        assert float(value) == pytest.approx(float(expected.detach()), abs=.005)
        actual = np.concatenate([np.array(gradients[name]).reshape(-1) for name in sorted(params)])
        expected_grad = dict(reference.named_parameters())
        target = np.concatenate([expected_grad[name].grad.numpy().reshape(-1) for name in sorted(params)])
        assert np.isfinite(actual).all()
        assert np.linalg.norm(actual - target) / np.linalg.norm(target) < .02
        assert np.dot(actual, target) / (np.linalg.norm(actual) * np.linalg.norm(target)) > .999
        assert all(p.dtype == mx.float32 for p in params.values())
        assert all(g.dtype == mx.float32 for g in gradients.values())
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_mlx_summed_bf16_backward_then_fp32_normalization(device):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    try:
        torch.manual_seed(782)
        reference = model()
        rows = torch.tensor([[2, 3, 7, 8, 0, 0, 0, 0], [2, 3, 4, 5, 6, 8, 9, 10]])
        x,y = rows[:,:-1],rows[:,1:]
        mask = target_mask(x,y,7,0)
        count = int(mask.sum())
        with torch.autocast('cpu',dtype=torch.bfloat16):
            summed = chunked_loss(reference,x,y,chunk_size=4,loss_mask=mask,reduction='sum')
        summed.backward()
        params,config = parameters_from_torch(reference),reference.config.to_dict()
        sizes = [t.num_embeddings for t in reference.model.engram.bigram_tables]
        sizes += [t.num_embeddings for t in reference.model.engram.trigram_tables]
        fn = scaled_value_and_grad(lambda p,batch,indices: loss(p,config,batch,indices,
            compute='bfloat16',query_tile=5,chunk=4,checkpoint=True,reduction='sum'),
            compute='bfloat16',compiled=True,loss_scale=1)
        value,gradients = fn(params,mx.array(rows.numpy()),memory_indices(config,x.numpy(),sizes))
        mx.eval(value,gradients)
        assert float(value)/count == pytest.approx(float(summed.detach())/count,abs=.005)
        actual = np.concatenate([np.array(gradients[k]/count).reshape(-1) for k in sorted(params)])
        names = dict(reference.named_parameters())
        expected = np.concatenate([(names[k].grad/count).numpy().reshape(-1) for k in sorted(params)])
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < .02
        assert np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)) > .999
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_bf16_silu_matches_fused_pytorch_opmath(device):
    previous=mx.default_device();mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
    try:
        rng=np.random.default_rng(741);x=rng.normal(size=(128,64)).astype(np.float32)*3
        upstream=rng.normal(size=x.shape).astype(np.float32)
        reference=torch.tensor(x,requires_grad=True)
        output=torch.nn.functional.silu(reference.to(torch.bfloat16))
        (output.float()*torch.tensor(upstream)).sum().backward()
        fn=mx.value_and_grad(lambda v:mx.sum(silu(v.astype(mx.bfloat16)).astype(mx.float32)*mx.array(upstream)))
        value,gradient=fn(mx.array(x));mx.eval(value,gradient)
        actual=np.array(gradient);expected=reference.grad.numpy()
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.001
    finally:mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_bf16_head64_grouped_attention_forward_and_backward(device):
    previous=mx.default_device();mx.set_default_device(mx.cpu if device=='cpu' else mx.gpu)
    try:
        rng=np.random.default_rng(492)
        values=[rng.normal(size=(1,h,128,64)).astype(np.float32) for h in [4,2,2]]
        upstream=rng.normal(size=values[0].shape).astype(np.float32)
        primals=[torch.tensor(x,requires_grad=True) for x in values]
        q,k,v=[x.to(torch.bfloat16) for x in primals]
        output=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,dim=1),
            v.repeat_interleave(2,dim=1),is_causal=True)
        (output.float()*torch.tensor(upstream)).sum().backward()
        fn=mx.value_and_grad(lambda q,k,v:mx.sum(attention(q.astype(mx.bfloat16),k.astype(mx.bfloat16),
            v.astype(mx.bfloat16),32).astype(mx.float32)*mx.array(upstream)),argnums=(0,1,2))
        value,grads=fn(*[mx.array(x) for x in values]);mx.eval(value,grads)
        for actual,expected in zip(grads,primals):
            a,b=np.array(actual),expected.grad.numpy()
            assert np.linalg.norm(a-b)/np.linalg.norm(b)<.01
    finally:mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_mlx_fp16_scaling_preserves_dense_vocabulary_gradients(device):
    import mlx.nn as nn
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    try:
        generator = np.random.default_rng(238)
        weights = generator.normal(0, .02, size=(65536, 4)).astype(np.float32)
        hidden = generator.normal(0, .1, size=(32, 4)).astype(np.float32)
        labels = np.arange(32, dtype=np.int32)
        reference = torch.tensor(weights, requires_grad=True)
        expected = torch.nn.functional.cross_entropy(torch.tensor(hidden) @ reference.T, torch.tensor(labels).long())
        expected.backward()
        h, y = mx.array(hidden), mx.array(labels)
        fn = scaled_value_and_grad(lambda w: nn.losses.cross_entropy(
            (h.astype(mx.float16) @ w.astype(mx.float16).T).astype(mx.float32), y, reduction='mean'),
            compute='float16', compiled=True)
        value, gradient = fn(mx.array(weights))
        mx.eval(value, gradient)
        actual, target = np.array(gradient), reference.grad.numpy()
        assert float(value) == pytest.approx(float(expected.detach()), abs=.002)
        assert np.isfinite(actual).all()
        assert np.linalg.norm(actual - target) / np.linalg.norm(target) < .003
        assert np.count_nonzero(actual[32:]) > actual[32:].size * .99
    finally:
        mx.set_default_device(previous)
