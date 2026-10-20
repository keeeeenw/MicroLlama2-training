"""Explicit next-token objective and token-weighted accumulation."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOSS_POLICY = "document-boundaries-v2"


def target_mask(x, y, eos_token_id, pad_token_id):
    """Learn document EOS, but not EOS -> unrelated next document or padding."""
    if x.shape != y.shape:
        raise ValueError("Inputs and next-token targets must have equal shapes")
    return (x != eos_token_id) & (x != pad_token_id) & (y != pad_token_id)


def chunked_loss(model, x, y, chunk_size=256, loss_mask=None, reduction="mean"):
    if chunk_size < 1 or reduction not in ("mean", "sum"):
        raise ValueError("Invalid chunk size or loss reduction")
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Inputs and targets must be equal [batch, sequence] tensors")
    if loss_mask is not None and loss_mask.shape != y.shape:
        raise ValueError("Loss mask shape differs from targets")
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
    total = hidden_nll(hidden, model.lm_head.weight, y, chunk_size, loss_mask)
    valid = ((y != -100) & loss_mask.bool()).sum() if loss_mask is not None else (y != -100).sum()
    if reduction == "mean" and int(valid) == 0:
        raise ValueError("Cannot average loss with no valid targets")
    return total / valid if reduction == "mean" else total


def hidden_nll(hidden, weight, y, chunk_size=256, loss_mask=None):
    """Summed primary/auxiliary NLL without materializing a sequence-wide vocabulary."""
    labels = y.reshape(-1).long()
    if loss_mask is not None:
        labels = labels.masked_fill(~loss_mask.reshape(-1).bool(), -100)
    hidden = hidden.reshape(-1, hidden.shape[-1])

    def part(h, w, target):
        return F.cross_entropy(F.linear(h, w).float(), target,
                               reduction="sum", ignore_index=-100)

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for i in range(0, len(labels), chunk_size):
        h, target = hidden[i:i + chunk_size], labels[i:i + chunk_size]
        if torch.is_grad_enabled():
            loss = checkpoint(part, h, weight, target,
                              use_reentrant=False)
        else:
            loss = part(h, weight, target)
        total = total + loss
    return total


def normalize_accumulated_gradients(model, valid_targets):
    """Apply one global normalization to summed microbatch NLL gradients."""
    if valid_targets < 1:
        raise ValueError("An optimizer update needs at least one valid target")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(valid_targets)
