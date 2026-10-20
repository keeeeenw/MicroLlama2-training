"""CUDA pretraining with FP32 master weights, BF16 compute, and resumable state."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from ..data_io.blocks import TokenBlocks, GrowingTokenBlocks, DataWaitStopped
from ..common.paths import configure_storage, checkpoint_root, training_log_root
from .objective import chunked_loss, target_mask, normalize_accumulated_gradients, LOSS_POLICY
from .continuation import validate_config, validate_handoff


def learning_rate(config, tokens):
    peak = config["learning_rate"]
    warmup = config["warmup_tokens"]
    if tokens < warmup:
        return peak * max(tokens, 1) / max(warmup, 1)
    if config.get("schedule") == "constant-after-warmup-v1":
        return peak
    ratio = min(1.0, (tokens - warmup) / max(1, config["target_tokens"] - warmup))
    floor = config["min_learning_rate"]
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * ratio))


def build_model(model_config, gradient_checkpointing=True):
    if model_config.get("model_type") in ("microllama_hybrid_engram", "microllama_dense_engram"):
        if gradient_checkpointing:
            raise ValueError("Hybrid/Engram decoder checkpointing is not qualified")
        from .hybrid import build_model as build_hybrid
        return build_hybrid(model_config)
    from transformers import Qwen3Config, Qwen3ForCausalLM
    c = Qwen3Config(**model_config)
    c.use_cache = False
    c._attn_implementation = "sdpa"
    # Construct randomly initialized weights; no pretrained model is loaded.
    model = Qwen3ForCausalLM(c).float()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def copy_tokenizer(root, destination):
    for name in ("tokenizer.json", "tokenizer_config.json", "source_manifest.json", "special_tokens_map.json", "chat_template.jinja"):
        src = root / "tokenizer" / name
        if src.exists():
            shutil.copy2(src, destination / name)


def save_checkpoint(root, run, model, optimizer, data, config, step, tokens, validation_shards=None, checkpoint_dir=None, extra_state=None):
    import torch
    target = (checkpoint_dir or root / "checkpoints") / run / f"step-{step:08d}"
    if target.exists():
        raise RuntimeError(f"Checkpoint already exists: {target}")
    temporary = target.with_name(target.name + ".partial")
    # A prior interrupted publication must be inspected, never merged into a new save.
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    copy_tokenizer(root, temporary)
    state = {"optimizer": optimizer.state_dict(), "data": data.state_dict(),
                "step": step, "tokens": tokens, "config": config, "validation_shards": validation_shards,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if extra_state:
        if set(extra_state) & set(state):
            raise ValueError("Extra restart state cannot overwrite core checkpoint fields")
        state.update(extra_state)
    torch.save(state, temporary / "training_state.pt")
    (temporary / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    import hashlib
    hashes = {}
    for path in sorted(temporary.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                os.fsync(f.fileno())
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    publication = temporary / "checkpoint_manifest.json"
    publication.write_text(json.dumps({"schema_version": 1, "step": step,
                          "tokens": tokens, "files": hashes}, indent=2) + "\n")
    with publication.open("rb") as f:
        os.fsync(f.fileno())
    sync_directory(temporary)
    temporary.rename(target)
    sync_directory(target.parent)
    latest = target.parent / "latest.json"
    temp_latest = latest.with_suffix(".tmp")
    temp_latest.write_text(json.dumps({"path": str(target), "step": step, "tokens": tokens}, indent=2) + "\n")
    with temp_latest.open("rb") as f:
        os.fsync(f.fileno())
    temp_latest.replace(latest)
    sync_directory(latest.parent)
    return target


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train-8k.json")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resume", type=Path)
    p.add_argument("--init-from", type=Path, help="Weights-only stage transition; fresh optimizer and schedule")
    p.add_argument("--continue-from", type=Path, help="Explicit same-context handoff preserving optimizer, RNG and data cursor")
    p.add_argument("--max-steps", type=int, help="Stop after this many total optimizer steps")
    p.add_argument("--allow-partial-data", action="store_true", help="Explicitly permit a prepared shard pilot")
    p.add_argument("--streaming-data", action="store_true", help="Consume immutable token chunks as they arrive")
    p.add_argument("--token-dir", type=Path, help="Local token staging directory")
    p.add_argument("--checkpoint-dir", type=Path, help="Fast checkpoint root; pipeline archives it to the data root")
    args = p.parse_args()
    if sum(bool(path) for path in (args.resume, args.init_from, args.continue_from)) > 1:
        p.error("Use one of --resume, --init-from or --continue-from")
    root = configure_storage()
    args.checkpoint_dir = args.checkpoint_dir or checkpoint_root(root)
    log_dir = training_log_root(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from safetensors.torch import load_model
    from torch.nn.attention import SDPBackend, sdpa_kernel
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    mc = json.loads(Path(config["model_config"]).read_text())
    seq = config["sequence_length"]
    if config["micro_batch_size"] < 1 or config["gradient_accumulation_steps"] < 1:
        raise RuntimeError("Batch and accumulation sizes must be positive")
    if seq > mc["max_position_embeddings"]:
        raise RuntimeError("Sequence length exceeds configured context")
    token_dir = args.token_dir or root / "tokens"
    prepared = json.loads((token_dir / "manifest.json").read_text())
    tokenizer_info = json.loads((root / "tokenizer" / "source_manifest.json").read_text())
    if prepared["tokenizer_sha256"] != tokenizer_info["tokenizer_sha256"]:
        raise RuntimeError("Prepared data tokenizer mismatch")
    if mc["vocab_size"] <= tokenizer_info["max_token_id"]:
        raise RuntimeError("Model vocabulary is smaller than the tokenizer")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if mc[key] != tokenizer_info[key]:
            raise RuntimeError(f"Special token mismatch: {key}")
    if not prepared["complete"] and not args.allow_partial_data and not args.streaming_data:
        raise RuntimeError("Dataset preparation incomplete; use --allow-partial-data only for a pilot")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process; check device permissions / sandbox")
    if config["precision"] == "bf16" and args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    torch.manual_seed(config["seed"])
    model = build_model(mc, config["gradient_checkpointing"]).to(args.device)
    model.generation_config.suppress_tokens = list(range(tokenizer_info["max_token_id"] + 1, mc["vocab_size"]))
    data = (GrowingTokenBlocks(token_dir, seq, seed=config["seed"]) if args.streaming_data
            else TokenBlocks(token_dir, seq, seed=config["seed"]))
    groups = [{"params": [v for v in model.parameters() if v.ndim >= 2], "weight_decay": config["weight_decay"]},
              {"params": [v for v in model.parameters() if v.ndim < 2], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], betas=tuple(config["betas"]),
                                 eps=1e-8, fused=args.device == "cuda")
    step = tokens = 0
    positions = 0
    restored_validation_shards = None
    lineage = None
    if args.resume or args.init_from or args.continue_from:
        path = args.resume or args.init_from or args.continue_from
        saved_mc = json.loads((path / "config.json").read_text())
        for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings"):
            if saved_mc[key] != mc[key]:
                raise RuntimeError(f"Checkpoint architecture mismatch: {key}")
        copied_tokenizer = path / "tokenizer.json"
        import hashlib
        if copied_tokenizer.exists() and hashlib.sha256(copied_tokenizer.read_bytes()).hexdigest() != prepared["tokenizer_sha256"]:
            raise RuntimeError("Checkpoint tokenizer mismatch")
        load_model(model, str(path / "model.safetensors"), strict=True, device=args.device)
        if args.resume or args.continue_from:
            state = torch.load(path / "training_state.pt", map_location=args.device, weights_only=True)
            if args.continue_from:
                changes = validate_handoff(state["config"], config, saved_mc, mc)
                lineage = {"parent_checkpoint": str(path.resolve()), "parent_step": state["step"],
                           "parent_tokens": state["tokens"], "changes": changes,
                           "state_preserved": ["optimizer", "rng", "data_cursor", "validation_inventory"],
                           "historical_target_policy": state["config"].get("loss_policy", "legacy-unmasked-v1"),
                           "counter_semantics": "Historical targets retain their original loss policy; new targets use the selected policy. No retroactive relabeling."}
            elif state["config"] != config:
                raise RuntimeError("Resume requires the same run configuration; use --continue-from for a validated same-context handoff")
            else:
                lineage = state.get("lineage")
            if saved_mc.get("max_position_embeddings") != mc["max_position_embeddings"] or saved_mc.get("rope_theta", 10000.0) != mc["rope_theta"]:
                raise RuntimeError("Resume context/RoPE mismatch")
            optimizer.load_state_dict(state["optimizer"])
            data.load_state_dict(state["data"])
            step, tokens = state["step"], state["tokens"]
            positions = state.get("input_positions", tokens)
            restored_validation_shards = state.get("validation_shards")
            torch.set_rng_state(state["torch_rng"].cpu())
            if args.device == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    autocast = lambda: torch.autocast(args.device, dtype=torch.bfloat16) if config["precision"] == "bf16" and args.device == "cuda" else nullcontext()
    attention = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if args.device == "cuda" else nullcontext()
    batch_tokens = config["micro_batch_size"] * seq * config["gradient_accumulation_steps"]
    stop_step = math.ceil(config["target_tokens"] / batch_tokens)
    if args.max_steps is not None:
        stop_step = min(stop_step, args.max_steps)
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "device": args.device,
                      "tokens_per_step": batch_tokens, "dataset_blocks": data.num_blocks,
                      "starting_step": step}), flush=True)
    log = log_dir / f"{config['run_name']}.jsonl"
    validation_path = log_dir / f"{config['run_name']}-validation-shards.json"
    if args.streaming_data:
        if restored_validation_shards is not None:
            validation_shards = restored_validation_shards
        elif validation_path.exists():
            validation_shards = json.loads(validation_path.read_text())
        else:
            validation_shards = [p.parent.name for p in sorted(token_dir.glob("*/manifest.json"))
                                 if not p.parent.name.endswith(".partial")
                                 and json.loads(p.read_text())["splits"]["validation"]["tokens"] > seq]
            if not validation_shards:
                raise RuntimeError("Wait for at least one validation token chunk before starting")
            validation_path.write_text(json.dumps(validation_shards) + "\n")
    else:
        validation_shards = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing this optimizer step and saving a resumable checkpoint.", flush=True)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    environment = {"torch": torch.__version__, "device": args.device,
                   "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                   "parameters_dtype": str(next(model.parameters()).dtype),
                   "attention_backends": ["flash", "efficient"] if args.device == "cuda" else ["math"],
                   "config": config, "model_config": mc, "data": data.state_dict(),
                   "tokenizer": tokenizer_info, "checkpoint_root": str(args.checkpoint_dir)}
    (log_dir / f"{config['run_name']}-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    last_saved = step if args.resume else -1
    model.train()
    try:
        while step < stop_step and not stop_requested:
            start = time.perf_counter()
            data_before_step = data.state_dict() if args.streaming_data else None
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            valid_sum = 0
            step_aborted = False
            for _ in range(config["gradient_accumulation_steps"]):
                try:
                    values = (data.next_batch(config["micro_batch_size"], lambda: stop_requested)
                              if args.streaming_data else data.next_batch(config["micro_batch_size"]))
                except DataWaitStopped:
                    data.load_state_dict(data_before_step)
                    step_aborted = True
                    break
                batch = torch.as_tensor(values, device=args.device)
                with autocast(), attention():
                    mask = (target_mask(batch[:, :-1], batch[:, 1:], mc["eos_token_id"], mc["pad_token_id"])
                            if config.get("loss_policy") == LOSS_POLICY else None)
                    valid = int(mask.sum()) if mask is not None else batch[:, 1:].numel()
                    loss = chunked_loss(model, batch[:, :-1], batch[:, 1:], config["loss_chunk_size"],
                                        loss_mask=mask, reduction="sum")
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; optimizer update skipped")
                loss.backward()
                loss_sum += loss.item()
                valid_sum += valid
            if step_aborted:
                optimizer.zero_grad(set_to_none=True)
                break
            normalize_accumulated_gradients(model, valid_sum)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            lr = learning_rate(config, tokens + valid_sum)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            tokens += valid_sum
            positions += batch_tokens
            if args.device == "cuda":
                torch.cuda.synchronize()
            record = {"step": step, "tokens": tokens, "loss": loss_sum / valid_sum,
                      "input_positions": positions, "valid_targets": valid_sum,
                      "valid_targets_since_handoff": tokens - lineage["parent_tokens"] if lineage else tokens,
                      "loss_policy": config.get("loss_policy", "legacy-unmasked-v1"),
                      "lr": lr, "grad_norm": float(norm), "tokens_per_second": valid_sum / (time.perf_counter() - start),
                      "data_epoch": data.epoch,
                      "corpus_train_tokens": data.available_tokens if args.streaming_data else prepared["train_tokens"],
                      "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None}
            if step % config["eval_every"] == 0:
                val = TokenBlocks(token_dir, seq, split="validation", seed=config["seed"], shards=validation_shards)
                model.eval()
                with torch.no_grad(), autocast(), attention():
                    nll = count = 0
                    for _ in range(config["eval_batches"]):
                        batch = torch.as_tensor(val.next_batch(config["micro_batch_size"]), device=args.device)
                        mask = (target_mask(batch[:, :-1], batch[:, 1:], mc["eos_token_id"], mc["pad_token_id"])
                                if config.get("loss_policy") == LOSS_POLICY else None)
                        nll += chunked_loss(model, batch[:, :-1], batch[:, 1:], config["loss_chunk_size"],
                                            loss_mask=mask, reduction="sum").item()
                        count += int(mask.sum()) if mask is not None else batch[:, 1:].numel()
                record["validation_loss"] = nll / count
                model.train()
            print(json.dumps(record), flush=True)
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if step % config["save_every"] == 0:
                print(f"Saved {save_checkpoint(root, config['run_name'], model, optimizer, data, config, step, tokens, validation_shards, args.checkpoint_dir, extra_state={'input_positions': positions, 'lineage': lineage})}", flush=True)
                last_saved = step
    except KeyboardInterrupt:
        # A partially accumulated step must not advance the checkpoint data cursor.
        raise RuntimeError("Interrupted during a step. Resume the last complete checkpoint.")
    if step != last_saved:
        print(f"Saved {save_checkpoint(root, config['run_name'], model, optimizer, data, config, step, tokens, validation_shards, args.checkpoint_dir, extra_state={'input_positions': positions, 'lineage': lineage})}", flush=True)
    print(json.dumps({"status": "stopped" if stop_requested else "completed", "step": step, "training_tokens": tokens}), flush=True)


if __name__ == "__main__":
    main()
