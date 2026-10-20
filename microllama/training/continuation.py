"""Deliberate same-context handoff; optimizer, RNG and sampler remain intact."""

import math

ALLOWED_CHANGES = {"run_name", "target_tokens", "learning_rate", "min_learning_rate",
                   "warmup_tokens", "loss_policy", "schedule", "eval_every",
                   "eval_batches", "save_every", "micro_batch_size", "gradient_accumulation_steps"}


def promote_confirmation_state(state, new, old_model, new_model):
    """Translate the bounded confirmation schema without resetting restart state.

    Branch counters exclude the shared dense parent. Add that historical exposure
    once for the full-run budget; Adam's own update counters remain untouched.
    Historical unmasked targets retain their original meaning in the lineage.
    This accepts only the current matched AdamW confirmation, never Muon or the
    legacy 100-update memory screen.
    """
    from .screen import branch_settings
    validate_config(new)
    old = state["config"]
    if old.get("profile") != "engram-confirmation" or old.get("branch") not in ("control", "engram"):
        raise ValueError("Promotion requires the revised Engram/control confirmation")
    if old != branch_settings(new, old["branch"], old["max_steps"], "engram-confirmation"):
        raise ValueError("Confirmation recipe differs from the selected continuation")
    if new.get("schedule") != "constant-after-warmup-v1" or new["learning_rate"] != old["adam_peak_lr"]:
        raise ValueError("Confirmation promotion must preserve the held backbone learning rate")
    if new["betas"] != old["betas"] or new["precision"] != "bf16" or new["gradient_checkpointing"]:
        raise ValueError("Confirmation promotion must preserve Adam betas and the qualified compute recipe")
    for key, value in new_model.items():
        if key not in ("model_type", "architectures") and old_model.get(key) != value:
            raise ValueError(f"Same-context continuation model mismatch: {key}")
    lineage = state.get("lineage", {})
    if not lineage.get("optimizer_reset") or not lineage.get("data_cursor_preserved"):
        raise ValueError("Confirmation checkpoint lacks its bound parent lineage")
    for key in ("parent_step", "parent_tokens"):
        if type(lineage.get(key)) is not int or lineage[key] < 0:
            raise ValueError("Invalid confirmation parent counters")
    shape = old["sequence_length"] * old["micro_batch_size"] * old["gradient_accumulation_steps"]
    if lineage.get("historical_loss_policy") != "legacy-unmasked-v1" or lineage["parent_tokens"] != lineage["parent_step"] * shape:
        raise ValueError("Confirmation promotion requires the bound legacy parent exposure")
    if not 0 < state["step"] <= old["max_steps"] or state["step"] < 51:
        raise ValueError("Confirmation checkpoint must have completed its short warmup")
    result = {**state, "step": lineage["parent_step"] + state["step"],
              "tokens": lineage["parent_tokens"] + state["tokens"],
              "input_positions": lineage["parent_tokens"] + state["input_positions"]}
    if not new["warmup_tokens"] <= result["tokens"] < new["target_tokens"]:
        raise ValueError("Continuation budget/warmup does not cover the promoted cursor")
    changes = {"source_profile": old["profile"], "branch": old["branch"],
               "branch_updates": state["step"], "branch_valid_targets": state["tokens"],
               "historical_parent_updates": lineage["parent_step"],
               "historical_parent_targets": lineage["parent_tokens"],
               "historical_parent_loss_policy": lineage.get("historical_loss_policy"),
               "counter_semantics": "Historical parent targets plus branch valid targets; no retroactive masking",
               "optimizer_and_cursor_preserved": True, "full_run_name": new["run_name"]}
    return result, changes


def validate_handoff(old, new, old_model, new_model):
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    unsupported = changed - ALLOWED_CHANGES
    if unsupported:
        raise ValueError(f"Unsupported continuation changes: {sorted(unsupported)}")
    # Repartition a fixed global batch for a larger-memory GPU; never silently
    # change optimizer batch size when moving between hardware profiles.
    if (old["micro_batch_size"] * old["gradient_accumulation_steps"] !=
            new["micro_batch_size"] * new["gradient_accumulation_steps"]):
        raise ValueError("Continuation must preserve the effective global batch")
    for key, value in new_model.items():
        if key in {"model_type", "architectures"}:
            continue
        if old_model.get(key) != value:
            raise ValueError(f"Same-context continuation model mismatch: {key}")
    if not changed or new["run_name"] == old["run_name"]:
        raise ValueError("A handoff requires a distinct run name and recorded changes")
    return {key: {"before": old.get(key), "after": new.get(key)} for key in sorted(changed)}


def validate_config(config):
    required = {"model_config", "run_name", "seed", "sequence_length", "micro_batch_size",
                "gradient_accumulation_steps", "target_tokens", "learning_rate",
                "min_learning_rate", "warmup_tokens", "weight_decay", "betas", "grad_clip",
                "precision", "gradient_checkpointing", "loss_chunk_size", "eval_every",
                "eval_batches", "save_every"}
    extra = set(config) - required - {"loss_policy", "schedule"}
    missing = required - set(config)
    if extra or missing:
        raise ValueError(f"Training config keys: unknown={sorted(extra)}, missing={sorted(missing)}")
    positive_ints = {"sequence_length", "micro_batch_size", "gradient_accumulation_steps",
                     "target_tokens", "loss_chunk_size", "eval_every", "eval_batches", "save_every"}
    for key in positive_ints:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["warmup_tokens"]) is not int or not 0 <= config["warmup_tokens"] < config["target_tokens"]:
        raise ValueError("warmup_tokens must be nonnegative and below target")
    if not 0 < config["min_learning_rate"] <= config["learning_rate"]:
        raise ValueError("Learning rates must be positive, with floor <= peak")
    for key in ("learning_rate", "min_learning_rate", "weight_decay", "grad_clip"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= config["weight_decay"] or not config["grad_clip"] > 0:
        raise ValueError("Invalid weight decay or gradient clipping")
    if len(config["betas"]) != 2 or not all(0 <= b < 1 for b in config["betas"]):
        raise ValueError("Invalid Adam betas")
    if config["precision"] not in ("bf16", "fp32") or type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("Unsupported precision or checkpointing setting")
    if config.get("schedule", "cosine-v1") not in ("cosine-v1", "constant-after-warmup-v1"):
        raise ValueError("Unknown learning-rate schedule")
    if config.get("loss_policy", "legacy-unmasked-v1") not in ("legacy-unmasked-v1", "document-boundaries-v2"):
        raise ValueError("Unknown loss policy")
