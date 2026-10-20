"""Bounded-memory causal likelihood helpers used by versioned evaluations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RollingWindow:
    """One scoring window; target indices are half-open indices into targets."""

    target_start: int
    target_end: int


def rolling_windows(target_count, context_length, stride):
    """Yield overlapping windows that score each next-token target once.

    ``context_length`` counts model input tokens. ``stride`` is smaller so
    each later scoring window retains the preceding text as context.
    """
    if target_count < 0:
        raise ValueError("target_count cannot be negative")
    if context_length < 2:
        raise ValueError("context_length must be at least two")
    if not 1 <= stride < context_length:
        raise ValueError("stride must be positive and smaller than context_length")
    for target_start in range(0, target_count, stride):
        target_end = min(target_start + stride, target_count)
        yield RollingWindow(target_start, target_end)


def target_nll_sum(model, conditioning_ids, target_ids, context_length, stride, *,
                   projection_chunk_size=128):
    """Return summed NLL in nats with bounded vocabulary-logit memory.

    ``conditioning_ids`` begins with BOS (and may include a prompt); target IDs
    follow it. Pass EOS as a target only for complete docs. Context preceding
    each scoring span is run through the model but excluded from its loss, so
    overlapping windows never count a target twice.
    """
    import torch
    import torch.nn.functional as F

    if not conditioning_ids:
        raise ValueError("conditioning_ids must include at least BOS")
    if projection_chunk_size < 1:
        raise ValueError("projection_chunk_size must be positive")
    full_sequence = list(conditioning_ids) + list(target_ids)
    context_prefix_length = len(conditioning_ids)
    total_nll = 0.0
    for window in rolling_windows(len(target_ids), context_length, stride):
        target_start = context_prefix_length + window.target_start
        target_end = context_prefix_length + window.target_end
        input_end = target_end - 1
        input_start = max(0, input_end - context_length)
        inputs = torch.tensor([full_sequence[input_start:input_end]], dtype=torch.long)
        target = torch.tensor(full_sequence[target_start:target_end], dtype=torch.long)
        hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0]
        logit_start = target_start - 1 - input_start
        logit_end = target_end - 1 - input_start
        selected = hidden[logit_start:logit_end]
        if selected.shape[0] != target.numel():
            raise RuntimeError("Rolling likelihood target/logit alignment failed")
        for offset in range(0, target.numel(), projection_chunk_size):
            end = min(offset + projection_chunk_size, target.numel())
            logits = F.linear(selected[offset:end], model.lm_head.weight).float()
            total_nll += F.cross_entropy(logits, target[offset:end], reduction="sum").item()
    return total_nll


def generation_suppression_ids(tokenizer_json, tokenizer_size, model_vocab_size, eos_token_id,
                               pad_token_id=None):
    """Return IDs forbidden in base-model generation while preserving EOS."""
    suppressed = set()
    if pad_token_id is not None:
        suppressed.add(int(pad_token_id))
    for item in tokenizer_json.get("added_tokens", []):
        if item.get("special"):
            suppressed.add(int(item["id"]))
        # Some tool/control markers in the reused vocabulary are marked
        # non-special by its source, but still imply unsupported behavior.
        content = item.get("content", "")
        if content.startswith("<|") and content.endswith("|>"):
            suppressed.add(int(item["id"]))
    suppressed.discard(int(eos_token_id))
    suppressed.update(range(int(tokenizer_size), int(model_vocab_size)))
    return sorted(i for i in suppressed if 0 <= i < model_vocab_size)
