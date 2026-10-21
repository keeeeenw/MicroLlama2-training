"""CPU-side checks for nontrivial semantics in the MPS qualification fixtures."""
import pytest
import torch

from scripts.mac.qualify_mps import close_metrics, hash_fixture, tiny_model
from microllama.training.engram import EngramState
from microllama.training.objective import chunked_loss, target_mask


def test_parity_gate_rejects_wrong_or_nonfinite_gradients():
    with pytest.raises(AssertionError):
        close_metrics(torch.tensor([1.0, 2.1]), torch.tensor([1.0, 2.0]), atol=1e-5, rtol=1e-4)
    with pytest.raises(AssertionError):
        close_metrics(torch.tensor([float("nan")]), torch.tensor([1.0]), atol=1e-5, rtol=1e-4)


def test_hash_fixture_covers_high_ids_aliases_and_document_boundaries():
    memory, ids, history = hash_fixture()
    assert {0, 7, 255, 256, 32767, 32768, 65535}.issubset(set(ids.flatten().tolist()))
    original = memory._indices(ids, history)
    alias = ids.clone()
    alias[alias == 13] = 12
    equivalent = memory._indices(alias, history)
    assert all(torch.equal(left, right) for left, right in zip(original, equivalent))
    assert original[0].shape == (2, 16, 4)
    assert not original[2][0, 6] and not original[2][0, 7]
    assert not original[3][0, 8]
    assert memory._next_state(ids, history).history_ids[0].tolist() == [-1, -1]


def test_tiny_model_uses_actual_revised_memory_and_boundary_loss_gradient():
    torch.manual_seed(9)
    model = tiny_model()
    assert model.config.engram_lookup_version == "normalized-multihead-v2"
    assert model.config.engram_hash_heads == 4
    assert model.lm_head.weight is model.model.embed_tokens.weight
    rows = torch.tensor([[2, 3, 4, 7, 5, 6, 8, 9, 0]])
    x, y = rows[:, :-1], rows[:, 1:]
    mask = target_mask(x, y, 7, 0)
    assert int(mask.sum()) == 6
    loss = chunked_loss(model, x, y, chunk_size=3, loss_mask=mask, reduction="sum")
    loss.backward()
    for name, parameter in model.model.engram.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0, name


def test_explicit_padding_adapter_preserves_forward_and_tied_head_gradients():
    from microllama.training.mac import install_mps_padding_adapter
    import copy
    torch.manual_seed(43)
    reference = tiny_model()
    with torch.no_grad():
        reference.model.embed_tokens.weight[0].fill_(.5)
    actual = copy.deepcopy(reference)
    install_mps_padding_adapter(actual)
    ids = torch.tensor([[0, 2, 3, 0, 5, 6, 8]])
    expected = reference(input_ids=ids, use_cache=False).logits
    observed = actual(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    observed.square().mean().backward()
    assert reference.lm_head.weight is reference.model.embed_tokens.weight
    assert actual.lm_head.weight is actual.model.embed_tokens.weight
    torch.testing.assert_close(actual.model.embed_tokens.weight.grad, reference.model.embed_tokens.weight.grad, atol=0, rtol=0)
    assert float(actual.model.embed_tokens.weight.grad[0].abs().sum()) > 0
