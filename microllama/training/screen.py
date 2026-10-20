"""Bounded, isolated CUDA warm-start screen from a saved main data cursor."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from ..common.paths import data_root
from ..data_io.blocks import GrowingTokenBlocks, TokenBlocks
from ..data_io.download import sha256
from ..data_io.transfer import atomic_json
from .cuda import save_checkpoint
from .objective import LOSS_POLICY, chunked_loss, target_mask, normalize_accumulated_gradients
from .optimizers import build_optimizer, set_schedule_fraction


def input_identity(checkpoint, token_dir):
    """Bind original restart payload and immutable chunk inventory to the screen."""
    state_files = {name: sha256(checkpoint / name)
                   for name in ("model.safetensors", "training_state.pt", "config.json",
                                "run_config.json", "tokenizer.json", "tokenizer_config.json")}
    chunks = {p.parent.name: sha256(p) for p in sorted(token_dir.glob("*/manifest.json"))
              if not p.parent.name.endswith(".partial")}
    if not chunks:
        raise ValueError("Screen requires completed prepared chunks")
    return {"checkpoint": str(checkpoint.resolve()), "files": state_files,
            "token_directory": str(token_dir.resolve()), "chunks": chunks,
            "summary_sha256": sha256(token_dir / "manifest.json")}


def branch_settings(parent, branch, steps, profile="screen"):
    if profile not in ("screen", "engram-confirmation"):
        raise ValueError("Unknown experiment profile")
    limit = 1000 if profile == "engram-confirmation" else 100
    allowed = ("control", "engram") if profile == "engram-confirmation" else ("control", "muon", "engram")
    if branch not in allowed or not 1 <= steps <= limit:
        raise ValueError("Invalid bounded branch")
    config = {"branch": branch, "seed": parent["seed"], "sequence_length": parent["sequence_length"],
            "micro_batch_size": parent["micro_batch_size"],
            "gradient_accumulation_steps": parent["gradient_accumulation_steps"],
            "loss_chunk_size": parent["loss_chunk_size"], "max_steps": steps,
            "loss_policy": LOSS_POLICY, "optimizer": "muon" if branch == "muon" else "adamw",
            "adam_peak_lr": 3e-4, "muon_peak_lr": 0.02, "betas": [0.9, 0.95],
            "weight_decay": parent["weight_decay"], "grad_clip": parent["grad_clip"],
            "warmup_valid_targets": 10 * parent["sequence_length"] * parent["micro_batch_size"] * parent["gradient_accumulation_steps"],
            "schedule": "linear-short-warmup-then-hold", "eval_batches": 8,
            "checkpoint_steps": [25, 50, 75, 100], "save_reserve_seconds": 120}
    if profile == "engram-confirmation":
        config.update(profile=profile, eval_batches=128, memory_lr_multiplier=5.0,
                      warmup_valid_targets=50 * parent["sequence_length"] * parent["micro_batch_size"] * parent["gradient_accumulation_steps"],
                      checkpoint_steps=[s for s in (250, 500, 750, 1000) if s <= steps],
                      evaluation_steps=[s for s in (5, 250, 500, 750, 1000) if s <= steps],
                      checkpoint_keep=2, memory_target_relative_rms=0.005,
                      minimum_memory_relative_rms=1e-5, initial_loss_delta_limit=0.02)
    return config


def validate_reader_identity(reader, expected, checkpoint_state):
    if reader.known != expected["chunks"]:
        raise RuntimeError("Screen chunk inventory differs from the frozen input identity")
    if checkpoint_state["mode"] != "growing":
        raise RuntimeError("Screen requires the saved growing-reader cursor")
    reader.load_state_dict(checkpoint_state)
    if reader.state_dict() != checkpoint_state:
        raise RuntimeError("Screen did not restore the exact saved data cursor")


def verify_validation_inventory(token_dir, shards, identity):
    """Check every frozen validation binary against its bound immutable manifest."""
    checksums = {}
    for name in shards:
        manifest = token_dir / name / "manifest.json"
        if name not in identity["chunks"] or sha256(manifest) != identity["chunks"][name]:
            raise RuntimeError(f"Frozen validation manifest changed: {name}")
        expected = json.loads(manifest.read_text())["splits"]["validation"]["sha256"]
        if sha256(manifest.parent / "validation.bin") != expected:
            raise RuntimeError(f"Frozen validation binary changed: {name}")
        checksums[name] = expected
    return checksums


def _residual_hook(model, ratios):
    def measure(module, inputs, output):
        hidden = inputs[0].float()
        difference = output[0].float() - hidden
        ratios.append(float((difference.square().mean().sqrt() /
                             hidden.square().mean().sqrt().clamp_min(1e-12)).cpu()))
    return model.model.engram.register_forward_hook(measure)


def calibrate_memory(model, input_ids, target_relative_rms):
    """One deterministic training-input calibration, without gradients or cursor consumption."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    was_training = model.training
    def measure():
        ratios = []
        hook = _residual_hook(model, ratios)
        try:
            model.eval()
            with torch.no_grad(), torch.autocast(input_ids.device.type, dtype=torch.bfloat16), sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                model.model(input_ids=input_ids, use_cache=False)
        finally:
            hook.remove()
            model.train(was_training)
        return sum(ratios) / len(ratios), ratios
    before, before_layers = measure()
    if not math_is_finite(before) or before <= 0:
        raise RuntimeError("Memory calibration requires a finite nonzero residual")
    gain = target_relative_rms / before
    if not 0.02 <= gain <= 100:
        raise RuntimeError("Memory calibration exceeds its bounded projection gain")
    with torch.no_grad():
        model.model.engram.value_proj.weight.mul_(gain)
    after, after_layers = measure()
    if not 0.8 * target_relative_rms <= after <= 1.2 * target_relative_rms:
        raise RuntimeError("Memory calibration missed the declared residual target")
    return {"target_relative_rms": target_relative_rms, "value_projection_gain": gain,
            "before_relative_rms": before, "after_relative_rms": after,
            "before_layer_ratios": before_layers, "after_layer_ratios": after_layers,
            "source": "first training block at saved cursor; cursor restored; no optimizer updates"}


def math_is_finite(value):
    import math
    return math.isfinite(value)


def memory_update_snapshot(model):
    """Sample rows for inexpensive actual optimizer-update measurements."""
    import torch
    return {name: (torch.arange(0, p.shape[0], max(1, p.shape[0] // 256), device=p.device)[:256],
                   p.detach()[::max(1, p.shape[0] // 256)][:256].clone())
            for name, p in model.named_parameters()
            if ".engram." in name and ("bigram_table" in name or "trigram_table" in name)}


def memory_update_metrics(model, before):
    metrics = {}
    parameters = dict(model.named_parameters())
    for name, (indices, values) in before.items():
        parameter = parameters[name]
        gradient_rms = float(parameter.grad.square().mean().sqrt())
        update_rms = float((parameter.detach()[indices] - values).square().mean().sqrt())
        if not math_is_finite(gradient_rms) or not math_is_finite(update_rms):
            raise RuntimeError(f"Nonfinite memory update: {name}")
        metrics[name] = {"gradient_rms": gradient_rms, "sampled_update_rms": update_rms}
    return metrics


def prune_checkpoints(directory, keep):
    """Retain the latest experiment publications; never touch partial, linked or main directories."""
    if keep < 1:
        raise ValueError("Checkpoint retention must keep a restart point")
    published = sorted(p for p in directory.glob("step-????????")
                       if p.is_dir() and not p.is_symlink() and p.name[5:].isdigit()
                       and (p / "checkpoint_manifest.json").is_file())
    removed = []
    for path in published[:-keep]:
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def validation(model, token_dir, config, shards, device, disable_memory=False):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    reader = TokenBlocks(token_dir, config["sequence_length"], split="validation",
                         seed=config["seed"], shards=shards)
    total = count = 0
    blocks, ratios = [], []
    revised = config.get("profile") == "engram-confirmation"
    memory = getattr(model.model, "engram", None)
    original_scale = memory.residual_scale.detach().clone() if disable_memory and memory is not None else None
    if revised and reader.num_blocks < config["eval_batches"] * config["micro_batch_size"]:
        raise RuntimeError("Larger validation must not cycle/repeat blocks")
    hook = _residual_hook(model, ratios) if revised and memory is not None and not disable_memory else None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16), sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if original_scale is not None:
                memory.residual_scale.zero_()
            for _ in range(config["eval_batches"]):
                indices = reader.order[reader.cursor:reader.cursor + config["micro_batch_size"]].tolist()
                batch = torch.as_tensor(reader.next_batch(config["micro_batch_size"]), device=device)
                x, y = batch[:, :-1], batch[:, 1:]
                mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
                nll = chunked_loss(model, x, y, config["loss_chunk_size"],
                                   loss_mask=mask, reduction="sum").item()
                valid = int(mask.sum())
                total += nll
                count += valid
                if revised:
                    blocks.append({"block_indices": indices, "nll": nll, "valid_targets": valid})
    finally:
        if hook is not None:
            hook.remove()
        if original_scale is not None:
            with torch.no_grad():
                memory.residual_scale.copy_(original_scale)
        model.train(was_training if revised else True)
    if count == 0:
        raise RuntimeError("Validation produced no valid targets")
    result = {"nll": total, "valid_targets": count, "loss": total / count,
            "protocol": "fixed-8k-document-boundaries-v2", "shards": shards}
    if revised:
        if count < 1_000_000:
            raise RuntimeError("Confirmation validation requires at least 1M valid targets")
        result.update(protocol="fixed-8k-128-block-document-boundaries-v2", blocks=blocks,
                      inventory_fingerprint=reader.fingerprint, seed=config["seed"])
        if ratios:
            result["memory_relative_rms"] = sum(ratios) / len(ratios)
            result["learned_residual_scale"] = float(memory.residual_scale.detach().cpu())
    return result


def evaluate_branch(model, token_dir, config, shards, device):
    measured = validation(model, token_dir, config, shards, device)
    if config.get("profile") == "engram-confirmation" and hasattr(model.model, "engram"):
        disabled = validation(model, token_dir, config, shards, device, disable_memory=True)
        measured["memory_disabled_loss"] = disabled["loss"]
        measured["disabled_minus_enabled_loss"] = disabled["loss"] - measured["loss"]
        measured["memory_disabled_blocks"] = disabled["blocks"]
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("control", "muon", "engram"), required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--profile", choices=("screen", "engram-confirmation"), default="screen")
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + args.wall_seconds
    directory = args.session_dir / args.branch
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    status = {"branch": args.branch, "phase": "loading", "pid": os.getpid(),
              "started_unix": time.time(), "max_wall_seconds": args.wall_seconds}
    atomic_json(status_path, status)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import Qwen3ForCausalLM
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The bounded screen needs BF16 CUDA")
    identity = json.loads((args.session_dir / "inputs.json").read_text())
    if str(args.checkpoint.resolve()) != identity["checkpoint"]:
        raise RuntimeError("Different parent checkpoint requested")
    for name, digest in identity["files"].items():
        if sha256(args.checkpoint / name) != digest:
            raise RuntimeError(f"Parent checkpoint changed: {name}")
    parent = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    parent.pop("optimizer")  # Deliberate fresh branch optimizer; main file is untouched.
    config = branch_settings(parent["config"], args.branch, args.max_steps, args.profile)
    if args.wall_seconds <= config["save_reserve_seconds"] + 30:
        raise ValueError("Wall budget must include loading and checkpoint reserve")
    atomic_json(directory / "config.json", config)
    reader = GrowingTokenBlocks(args.token_dir, config["sequence_length"], seed=config["seed"])
    validate_reader_identity(reader, identity, parent["data"])
    initial_cursor = reader.state_dict()
    shards = parent.get("validation_shards")
    if not shards:
        raise RuntimeError("Parent lacks its frozen validation inventory")
    validation_checksums = verify_validation_inventory(args.token_dir, shards, identity)
    torch.manual_seed(config["seed"])
    dense = Qwen3ForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation="sdpa")
    if (dense.config.max_position_embeddings < config["sequence_length"] or
            dense.config.eos_token_id != 7 or dense.config.pad_token_id != 0 or
            dense.config.bos_token_id != 1):
        raise RuntimeError("Checkpoint context or pinned Liquid special IDs differ")
    if args.branch == "engram":
        from .engram import EngramQwen3ForCausalLM, LOOKUP_V2, build_token_id_map
        if args.profile == "engram-confirmation":
            model = EngramQwen3ForCausalLM.from_dense(dense, residual_scale=1.0,
                lookup_version=LOOKUP_V2, hash_heads=4,
                token_id_map=build_token_id_map(args.checkpoint / "tokenizer.json", dense.config.vocab_size))
        else:
            model = EngramQwen3ForCausalLM.from_dense(dense)
        del dense
        gc.collect()
    else:
        model = dense
    model.config.use_cache = False
    model.config._attn_implementation = "sdpa"
    model = model.float().to("cuda")
    model.train()
    calibration = None
    if args.profile == "engram-confirmation" and args.branch == "engram" and not args.resume:
        try:
            reference = reader.next_batch(1)
            reference_path = args.token_dir / reader.current / "train.bin"
            if sha256(reference_path) != reader.metadata[reader.current]["splits"]["train"]["sha256"]:
                raise RuntimeError("Calibration input checksum failed")
            calibration = calibrate_memory(model, torch.as_tensor(reference[:, :-1], device="cuda"),
                                           config["memory_target_relative_rms"])
        finally:
            reader.load_state_dict(initial_cursor)
        if reader.state_dict() != initial_cursor:
            raise RuntimeError("Calibration changed the saved training cursor")
    optimizer = build_optimizer(model, config["optimizer"], config["adam_peak_lr"],
                  config["muon_peak_lr"], config["weight_decay"], fused=True,
                  memory_lr_multiplier=config.get("memory_lr_multiplier", 1.0))
    step = valid_total = positions = 0
    torch.set_rng_state(parent["torch_rng"])
    torch.cuda.set_rng_state_all(parent["cuda_rng"])
    lineage = {"parent_checkpoint": identity["checkpoint"], "parent_step": parent["step"],
               "parent_tokens": parent["tokens"], "parent_files": identity["files"],
               "optimizer_reset": True, "data_cursor_preserved": True,
               "initial_cursor": initial_cursor, "historical_loss_policy": "legacy-unmasked-v1"}
    if args.resume:
        resume_state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        if resume_state["config"] != config or resume_state["lineage"] != lineage:
            raise RuntimeError("Branch resume identity/config mismatch")
        if args.branch == "engram":
            from .engram import load_engram_model
            del model, optimizer
            torch.cuda.empty_cache()
            model = load_engram_model(args.resume).float().to("cuda").train()
        else:
            from safetensors.torch import load_model
            load_model(model, str(args.resume / "model.safetensors"), strict=True, device="cuda")
        optimizer = build_optimizer(model, config["optimizer"], config["adam_peak_lr"],
                    config["muon_peak_lr"], config["weight_decay"], fused=True,
                    memory_lr_multiplier=config.get("memory_lr_multiplier", 1.0))
        optimizer.load_state_dict(resume_state["optimizer"])
        reader.load_state_dict(resume_state["data"])
        step, valid_total = resume_state["step"], resume_state["tokens"]
        positions = resume_state["input_positions"]
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in resume_state["cuda_rng"]])
        calibration = resume_state.get("provenance", {}).get("memory_calibration")
        del resume_state
    provenance = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
                  "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "loss_policy": LOSS_POLICY, "chunk_inventory_sha256": sha256(args.session_dir / "inputs.json"),
                  "muon_source": json.loads((Path(__file__).parent / "vendor/muon/source.json").read_text()),
                  "model_parameters": sum(p.numel() for p in model.parameters()),
                  "attention_backends": ["flash", "efficient"], "master_dtype": "float32"}
    provenance["validation_binary_checksums"] = validation_checksums
    if args.branch == "engram":
        provenance["engram"] = model.engram_metadata()
        if calibration:
            provenance["memory_calibration"] = calibration
    atomic_json(directory / "provenance.json", provenance)
    torch.cuda.reset_peak_memory_stats()
    initial_validation = evaluate_branch(model, args.token_dir, config, shards, "cuda")
    if calibration and abs(initial_validation["disabled_minus_enabled_loss"]) > config["initial_loss_delta_limit"]:
        raise RuntimeError("Calibrated memory perturbs initial validation beyond its declared limit")
    history = [{"step": step, "valid_targets": valid_total, "validation": initial_validation}]
    run = None
    def tracking_log(values):
        if run and not status.get("tracking_disabled"):
            try:
                run.log(values)
            except Exception as error:
                status.update(tracking_disabled=True,
                              wandb_error=f"{type(error).__name__}: {error}")
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project="MicroLlama2", name=f"{args.session_dir.name}-{args.branch}",
                    group=args.session_dir.name, job_type="warm-start-screen", dir=str(directory.resolve()),
                    config={**config, **provenance, "lineage": lineage},
                    settings=wandb.Settings(disable_git=True, disable_code=True, console="off", init_timeout=30))
            status["wandb_url"] = run.url
            tracking_log({"step": step, "validation/loss": initial_validation["loss"]})
        except Exception as error:
            status["wandb_error"] = f"{type(error).__name__}: {error}"
    status.update(phase="training", config=config, initial_validation=initial_validation,
                  model_parameters=provenance["model_parameters"])
    atomic_json(status_path, status)
    last_saved = step if args.resume else -1
    verified_files = set()
    try:
        while step < args.max_steps and not stopped:
            if time.monotonic() + config["save_reserve_seconds"] + 20 >= deadline:
                status["stop_reason"] = "time_budget"
                break
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            nll = valid = 0
            for _ in range(config["gradient_accumulation_steps"]):
                values = reader.next_batch(config["micro_batch_size"])
                if reader.current not in verified_files:
                    token_file = args.token_dir / reader.current / "train.bin"
                    if sha256(token_file) != reader.metadata[reader.current]["splits"]["train"]["sha256"]:
                        raise RuntimeError(f"Token chunk content checksum failed: {reader.current}")
                    verified_files.add(reader.current)
                batch = torch.as_tensor(values, device="cuda")
                x, y = batch[:, :-1], batch[:, 1:]
                mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
                count = int(mask.sum())
                with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    loss = chunked_loss(model, x, y, config["loss_chunk_size"], loss_mask=mask, reduction="sum")
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite branch loss")
                loss.backward()
                nll += loss.item()
                valid += count
            normalize_accumulated_gradients(model, valid)
            if step == 0:
                for name, parameter in model.named_parameters():
                    if parameter.dtype != torch.float32 or parameter.grad is None or parameter.grad.dtype != torch.float32:
                        raise RuntimeError(f"Missing FP32 gradient/master parameter: {name}")
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Nonfinite gradient: {name}")
                    if ".engram." in name and parameter.grad.abs().sum().item() == 0:
                        raise RuntimeError(f"Inactive memory gradient: {name}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            fraction = min(1.0, (valid_total + valid) / config["warmup_valid_targets"])
            set_schedule_fraction(optimizer, fraction)
            before = memory_update_snapshot(model) if args.branch == "engram" and args.profile == "engram-confirmation" and (step < 5 or (step + 1) % 100 == 0) else {}
            optimizer.step()
            updates = memory_update_metrics(model, before) if before else {}
            step += 1
            valid_total += valid
            positions += values.shape[0] * config["sequence_length"] * config["gradient_accumulation_steps"]
            torch.cuda.synchronize()
            record = {"step": step, "valid_targets": valid_total, "input_positions": positions,
                      "loss": nll / valid, "grad_norm": float(norm),
                      "valid_targets_per_second": valid / (time.monotonic() - tick),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                      "schedule_fraction": fraction, "elapsed_seconds": time.monotonic() - started}
            if updates:
                record["memory_table_updates"] = updates
                if any(m["gradient_rms"] <= 0 or m["sampled_update_rms"] <= 0 for m in updates.values()):
                    raise RuntimeError("Memory table has no measured gradient or optimizer update")
                record["memory/min_table_update_rms"] = min(m["sampled_update_rms"] for m in updates.values())
            with (directory / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            tracking_log(record)
            status.update(last_step=step, valid_targets=valid_total, elapsed_seconds=record["elapsed_seconds"])
            atomic_json(status_path, status)
            if step in config.get("evaluation_steps", config["checkpoint_steps"]):
                measured = evaluate_branch(model, args.token_dir, config, shards, "cuda")
                history.append({"step": step, "valid_targets": valid_total, "validation": measured})
                status["validation_history"] = history
                diagnostic = {key: measured[key] for key in ("memory_relative_rms", "memory_disabled_loss", "disabled_minus_enabled_loss", "learned_residual_scale") if key in measured}
                tracking_log({"step": step, "validation/loss": measured["loss"], "valid_targets": valid_total, **diagnostic})
                if "memory_relative_rms" in measured:
                    if measured["memory_relative_rms"] < config["minimum_memory_relative_rms"] or measured["memory_relative_rms"] > 0.05:
                        raise RuntimeError("Memory integration failed its active/stable residual gate")
                    if step == 5:
                        status.update(qualification_passed=True, qualification_step=step,
                                      qualification_memory=diagnostic)
                atomic_json(status_path, status)
            if step in config["checkpoint_steps"]:
                checkpoint = save_checkpoint(data_root(), args.branch, model, optimizer, reader, config,
                    step, valid_total, shards, args.session_dir / "checkpoints",
                    extra_state={"input_positions": positions, "lineage": lineage, "provenance": provenance})
                status["checkpoint"] = str(checkpoint)
                last_saved = step
                if config.get("checkpoint_keep"):
                    status.setdefault("pruned_checkpoints", []).extend(prune_checkpoints(checkpoint.parent, config["checkpoint_keep"]))
        if step != last_saved:
            measured = evaluate_branch(model, args.token_dir, config, shards, "cuda")
            history.append({"step": step, "valid_targets": valid_total, "validation": measured})
            checkpoint = save_checkpoint(data_root(), args.branch, model, optimizer, reader, config,
                step, valid_total, shards, args.session_dir / "checkpoints",
                extra_state={"input_positions": positions, "lineage": lineage, "provenance": provenance})
            status["checkpoint"] = str(checkpoint)
            if config.get("checkpoint_keep"):
                status.setdefault("pruned_checkpoints", []).extend(prune_checkpoints(checkpoint.parent, config["checkpoint_keep"]))
        status.update(phase="completed" if step == args.max_steps else "stopped",
                      last_step=step, valid_targets=valid_total, input_positions=positions,
                      validation_history=history, elapsed_seconds=time.monotonic() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        atomic_json(status_path, status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        status.update(phase="failed", error=f"{type(error).__name__}: {error}",
                      last_step=step, elapsed_seconds=time.monotonic()-started)
        atomic_json(status_path, status)
        raise
    finally:
        if run:
            try:
                run.finish(exit_code=0 if status.get("phase") in ("completed", "stopped") else 1)
            except Exception as error:
                status.update(tracking_disabled=True, wandb_error=f"{type(error).__name__}: {error}")
            atomic_json(status_path, status)


if __name__ == "__main__":
    main()
