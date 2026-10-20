import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from microllama.training.objective import chunked_loss, target_mask, normalize_accumulated_gradients
from microllama.training.cuda import build_model
from test_training import TINY


def test_document_eos_is_learned_but_next_document_transition_is_not():
    x = torch.tensor([[3, 4, 7, 5, 6, 7, 0]])
    y = torch.tensor([[4, 7, 5, 6, 7, 0, 0]])
    assert target_mask(x, y, 7, 0).tolist() == [[True, True, False, True, True, False, False]]


def test_masked_chunked_loss_and_gradients_match_full_logits():
    torch.manual_seed(8)
    a = build_model(TINY, gradient_checkpointing=False)
    b = copy.deepcopy(a)
    x = torch.tensor([[3, 4, 7, 5, 6, 7], [8, 7, 9, 10, 11, 0]])
    y = torch.tensor([[4, 7, 5, 6, 7, 0], [7, 9, 10, 11, 0, 0]])
    mask = target_mask(x, y, 7, 0)
    chunk = chunked_loss(a, x, y, 3, loss_mask=mask)
    labels = y.masked_fill(~mask, -100)
    full = torch.nn.functional.cross_entropy(b(x).logits.reshape(-1, 64), labels.reshape(-1))
    torch.testing.assert_close(chunk, full)
    chunk.backward()
    full.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=1e-4)


def test_different_valid_lengths_accumulate_by_targets_not_microbatch_means():
    torch.manual_seed(5)
    a = build_model(TINY, gradient_checkpointing=False)
    b = copy.deepcopy(a)
    x = torch.tensor([[2, 3, 7, 4, 5], [7, 9, 0, 0, 0]])
    y = torch.tensor([[3, 7, 4, 5, 7], [9, 0, 0, 0, 0]])
    mask = target_mask(x, y, 7, 0)
    # Second row has no loss; it must not create NaN or dilute the first row.
    for i in range(2):
        chunked_loss(a, x[i:i+1], y[i:i+1], 2, loss_mask=mask[i:i+1], reduction="sum").backward()
    normalize_accumulated_gradients(a, int(mask.sum()))
    chunked_loss(b, x, y, 2, loss_mask=mask).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=1e-4)
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-3)
    oa.step()
    ob.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=1e-4)


def test_empty_mean_rejected_and_mask_shape_checked():
    model = build_model(TINY, gradient_checkpointing=False)
    x = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="no valid"):
        chunked_loss(model, x, x, loss_mask=torch.zeros_like(x, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        chunked_loss(model, x, x, loss_mask=torch.ones((1, 1), dtype=torch.bool))
