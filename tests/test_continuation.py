import copy
import json
from pathlib import Path

import pytest

from microllama.training.continuation import validate_config, validate_handoff, promote_confirmation_state


def test_explicit_same_context_handoff_permits_only_declared_changes():
    old = json.loads(Path("configs/train-8k.json").read_text())
    model = json.loads(Path("configs/model-338m.json").read_text())
    new = {**old, "run_name": "general-8k-v2", "loss_policy": "document-boundaries-v2",
           "schedule": "constant-after-warmup-v1", "target_tokens": 44_000_000_000}
    validate_config(new)
    changes = validate_handoff(old, new, model, model)
    assert changes["loss_policy"]["after"] == "document-boundaries-v2"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_handoff(old, {**new, "betas": [0.8, 0.95]}, model, model)
    with pytest.raises(ValueError, match="model mismatch"):
        validate_handoff(old, new, model, {**model, "rope_theta": 100000})
    with pytest.raises(ValueError, match="Unsupported"):
        validate_handoff(old, {**new, "sequence_length": 16384}, model, model)


def test_hardware_handoff_repartitions_batch_without_changing_optimizer_batch():
    old = json.loads(Path("configs/train-8k-local.json").read_text())
    model = json.loads(Path("configs/model-338m.json").read_text())
    new = {**old, "run_name": "cloud-48gb", "micro_batch_size": 2, "gradient_accumulation_steps": 8}
    changes = validate_handoff(old, new, model, model)
    assert changes["micro_batch_size"] == {"before": 1, "after": 2}
    assert changes["gradient_accumulation_steps"] == {"before": 16, "after": 8}
    with pytest.raises(ValueError, match="effective global batch"):
        validate_handoff(old, {**new, "gradient_accumulation_steps": 16}, model, model)


@pytest.mark.parametrize("key,value", [("unknown", 1), ("precision", "fp8"),
    ("betas", [0.9, 1.0]), ("micro_batch_size", 0), ("warmup_tokens", -1),
    ("gradient_checkpointing", "false"), ("loss_policy", "magic")])
def test_invalid_config_rejected(key, value):
    config = json.loads(Path("configs/train-8k.json").read_text())
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def confirmation_fixture(branch="control"):
    from microllama.training.screen import branch_settings
    config = json.loads(Path("configs/train-8k-local.json").read_text())
    model = json.loads(Path("configs/model-338m.json").read_text())
    state = {"config": branch_settings(config, branch, 1000, "engram-confirmation"),
             "step": 1000, "tokens": 130942798, "input_positions": 131072000,
             "optimizer": {"step": 1000}, "data": {"cursor": 41},
             "torch_rng": object(), "cuda_rng": object(), "validation_shards": ["frozen"],
             "lineage": {"optimizer_reset": True, "data_cursor_preserved": True,
                         "parent_step": 1956, "parent_tokens": 256376832,
                         "historical_loss_policy": "legacy-unmasked-v1"}}
    return state, config, model


@pytest.mark.parametrize("branch", ["control", "engram"])
def test_confirmation_promotion_counts_parent_once_and_preserves_restart_payload(branch):
    state, config, model = confirmation_fixture(branch)
    result, changes = promote_confirmation_state(state, config, model, model)
    assert result["step"] == 2956 and result["tokens"] == 387319630
    assert result["input_positions"] == 387448832
    for key in ("optimizer", "data", "torch_rng", "cuda_rng", "validation_shards"):
        assert result[key] is state[key]
    assert state["step"] == 1000 and state["tokens"] == 130942798
    assert changes["branch"] == branch and changes["optimizer_and_cursor_preserved"]
    with pytest.raises(ValueError, match="requires the revised"):
        promote_confirmation_state({**result, "config": config}, config, model, model)


@pytest.mark.parametrize("key,value", [("betas", [0.8, 0.95]), ("seed", 1),
    ("sequence_length", 4096), ("learning_rate", 0.0001), ("schedule", "cosine-v1"),
    ("gradient_checkpointing", True), ("precision", "fp32"), ("weight_decay", 0.2),
    ("grad_clip", 0.5)])
def test_confirmation_promotion_rejects_recipe_changes(key, value):
    state, config, model = confirmation_fixture()
    with pytest.raises(ValueError):
        promote_confirmation_state(state, {**config, key: value}, model, model)


def test_confirmation_promotion_rejects_incomplete_lineage_and_context_changes():
    state, config, model = confirmation_fixture()
    with pytest.raises(ValueError, match="lineage"):
        promote_confirmation_state({**state, "lineage": {}}, config, model, model)
    with pytest.raises(ValueError, match="model mismatch"):
        promote_confirmation_state(state, config, model, {**model, "rope_theta": 999})
    with pytest.raises(ValueError, match="warmup"):
        promote_confirmation_state({**state, "step": 5}, config, model, model)
    with pytest.raises(ValueError, match="legacy parent"):
        promote_confirmation_state({**state, "lineage": {**state["lineage"], "parent_tokens": 1}}, config, model, model)
