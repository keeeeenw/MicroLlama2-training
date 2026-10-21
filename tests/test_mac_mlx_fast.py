"""Native Mac graph: independent objective/derivative and scheduling checks."""
import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training.mac_mlx import parameters_from_torch, memory_indices, scaled_value_and_grad
from microllama.training.mac_mlx_fast import loss, FastEngine
from microllama.training.objective import chunked_loss, target_mask
from test_mac_training import model
from test_mac_ultra_run import fixture


@pytest.fixture(autouse=True)
def restore_device():
    previous = mx.default_device()
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('gpu_embedding', [False, True])
def test_native_bf16_graph_matches_independent_torch_objective_and_all_gradients(device, checkpoint, fused, gpu_embedding):
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    torch.manual_seed(619)
    reference = model()
    rows = torch.tensor([[0, 2, 3, 3, 7, 8, 9, 0], [3, 3, 4, 5, 6, 8, 9, 10]])
    x, y = rows[:, :-1], rows[:, 1:]
    count = int(target_mask(x, y, 7, 0).sum())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = chunked_loss(reference, x, y, chunk_size=4,
                                loss_mask=target_mask(x, y, 7, 0), reduction='sum')
    expected.backward()
    params, config = parameters_from_torch(reference), reference.config.to_dict()
    sizes = [table.num_embeddings for table in reference.model.engram.bigram_tables]
    sizes += [table.num_embeddings for table in reference.model.engram.trigram_tables]
    fn = scaled_value_and_grad(lambda p, batch, indices: loss(p, config, batch, indices,
        compute='bfloat16', chunk=4, checkpoint=checkpoint, fused_projections=fused, gpu_embedding=gpu_embedding),
        compute='bfloat16', compiled=True, loss_scale=1)
    value, gradients = fn(params, mx.array(rows.numpy()), memory_indices(config, x.numpy(), sizes))
    mx.eval(value, gradients)
    assert float(value)/count == pytest.approx(float(expected.detach())/count, abs=.005)
    actual = np.concatenate([np.asarray(gradients[n]).reshape(-1) for n in sorted(params)])
    named = dict(reference.named_parameters())
    target = np.concatenate([named[n].grad.numpy().reshape(-1) for n in sorted(params)])
    assert np.linalg.norm(actual-target)/np.linalg.norm(target) < .025
    assert np.dot(actual, target)/(np.linalg.norm(actual)*np.linalg.norm(target)) > .999
    assert all(g.dtype == mx.float32 for g in gradients.values())
    assert np.abs(np.asarray(gradients['model.embed_tokens.weight'])[0]).sum() > 0


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_bounded_async_schedule_preserves_exact_global_gradient_and_adam_update(device):
    from microllama.training.mac_mlx import adam_update
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    initial, rows, _ = fixture(1)
    a = FastEngine(initial.params, initial.slots, initial.policy, initial.config, chunk=4)
    b = FastEngine(initial.params, initial.slots, initial.policy, initial.config, chunk=4, accumulation_window=2)
    av, ag, ac = a.gradients(rows)
    bv, bg, bc = b.gradients(rows)
    assert av == pytest.approx(bv, abs=1e-6)
    assert ac == bc
    for name in ag:
        np.testing.assert_array_equal(np.asarray(ag[name]), np.asarray(bg[name]), err_msg=name)
    ap, ast, _ = adam_update(a.params, ag, a.slots, a.policy)
    bp, bst, _ = adam_update(b.params, bg, b.slots, b.policy)
    mx.eval(ap, ast, bp, bst)
    for name in ap:
        np.testing.assert_array_equal(np.asarray(ap[name]), np.asarray(bp[name]), err_msg=name)
        for slot in ('m', 'v'):
            np.testing.assert_array_equal(np.asarray(ast[name][slot]), np.asarray(bst[name][slot]), err_msg=name)


def test_unbounded_evaluation_window_is_refused():
    with pytest.raises(ValueError, match='bounded'):
        FastEngine({}, {}, {}, {}, accumulation_window=16)
