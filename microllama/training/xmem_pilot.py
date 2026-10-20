"""Bounded frozen-backbone memory pilots; saves reader state, never a full model.

Run this module once per fresh session. The coordinator holds the project's GPU
lock and runs matched pretrained/shuffled conditions serially. XMemTransfer is
the default; --memory-kind graft uses offline teacher exact-suffix features. See
docs/experiments.md for the protocol, provenance, resume and interpretation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..data_io.blocks import GrowingTokenBlocks, TokenBlocks
from ..data_io.download import sha256
from ..data_io.transfer import atomic_json
from ..evaluation.likelihood import generation_suppression_ids
from .engram import load_engram_model
from .gpu import gpu_lock, idle_gpu
from .objective import chunked_loss, normalize_accumulated_gradients, target_mask
from .screen import input_identity, verify_validation_inventory
from .xmem import FrozenMemory, TargetReader, WordBoundaryAddressing


CONDITIONS = ("pretrained", "shuffled")
PROMPTS = ("The capital of France is", "Water freezes at", "A triangle has", "To make a cup of tea,")


def pilot_limits(memory_kind, profile):
    """Keep the short screen fixed; opt into a bounded longer graft comparison."""
    if profile == "graft-confirmation":
        if memory_kind != "graft":
            raise ValueError("The graft-confirmation profile requires --memory-kind graft")
        return 2048, 14400
    if profile != "pilot":
        raise ValueError("Unknown reader experiment profile")
    return (512 if memory_kind == "graft" else 256), 3600


def evaluation_steps(max_steps, profile):
    """Publish matched intermediate checks and always evaluate the final update."""
    checkpoints = (64, 128, 256, 512, 1024, 1536) if profile == "graft-confirmation" else (64, 128, 256)
    return {n for n in (*checkpoints, max_steps) if n <= max_steps}


def select_training_cursor(parent, source):
    """Select an explicit compatible cursor without mutating production state.

    A mixture checkpoint retains its last original FineWeb cursor. Replaying
    that saved cursor is a local adapter test, never a mixture continuation.
    Missing or incompatible inputs must fail rather than reset the reader.
    """
    state = parent["data"]
    if source == "retired-fineweb":
        if state.get("mode") != "mixture-growth-v1" or "retired_reader" not in state:
            raise ValueError("Retired FineWeb replay requires a mixture checkpoint with a saved retired_reader")
        state = state["retired_reader"]
    elif source != "parent":
        raise ValueError("Unknown pilot training data source")
    if state.get("mode") != "growing":
        raise ValueError("Pilot requires a growing cursor; explicitly select --training-data-source retired-fineweb for a mixture checkpoint")
    if state["seed"] != parent["config"]["seed"] or state["sequence_length"] != parent["config"]["sequence_length"]:
        raise ValueError("Selected pilot cursor differs from the parent seed or context length")
    return copy.deepcopy(state)


class ReaderInjection:
    """Add one external reader immediately after the parent's first native Engram.

    Precomputed addresses are detached input features; gradients must flow from
    the reader through the subsequent frozen layers. The parent owns no new
    parameters, and disabling this hook reproduces its original forward.
    """
    def __init__(self, model, reader, memory, addressing, layer=4):
        if not 0 < layer < len(model.model.layers):
            raise ValueError("Reader pre-hook must target a valid later decoder layer")
        self.reader, self.bank, self.addressing = reader, memory, addressing
        self.enabled = True
        self.memory = self.active = None
        self.calibration_target = None
        self.measure = False
        self.metrics = {}
        self.handle = model.model.layers[layer].register_forward_pre_hook(self._hook)

    def batch(self, input_ids):
        cpu_ids = input_ids.cpu()
        indices, active = self.addressing.encode_with_mask(cpu_ids)
        device = next(self.reader.parameters()).device
        with torch.no_grad():
            self.memory = self.bank(indices.to(device))
        self.active = active.to(device)
        if hasattr(self.reader, "boundary_mask"):
            controls = torch.tensor(sorted(self.addressing.protected), dtype=torch.long)
            self.reader.boundary_mask = torch.isin(cpu_ids, controls).to(device)
        self.metrics = {"active_fraction": float(active.float().mean())}
        if getattr(self.addressing, "last_orders", None) is not None:
            self.metrics.update({f"matched_{n}gram_fraction": float(self.addressing.last_orders.eq(n).float().mean())
                                 for n in (2, 3, 4)})

    def _hook(self, module, inputs):
        if not self.enabled:
            return None
        hidden = inputs[0]
        if self.memory is None or self.memory.shape[:-1] != hidden.shape[:-1]:
            raise RuntimeError("External memory must be prepared for this exact forward batch")
        if self.calibration_target is not None:
            self.reader.calibrate_residual_scale(hidden.detach(), self.memory, self.active,
                                                  target_ratio=self.calibration_target)
        delta, gates = self.reader(hidden, self.memory, self.active)
        if self.measure or self.calibration_target is not None:
            self.metrics.update(relative_rms=float((delta.detach().float().square().mean().sqrt() /
                hidden.detach().float().square().mean().sqrt().clamp_min(1e-12))),
                gate_mean=float(gates.detach().float().mean()),
                residual_scale=float(self.reader.residual_scale))
        return (hidden + delta,) + inputs[1:]

    def close(self):
        self.handle.remove()
        self.memory = self.active = None
        if hasattr(self.reader, "boundary_mask"):
            self.reader.boundary_mask = None


def tensor_digest(module):
    """Byte-level in-memory identity, including buffers, in bounded CPU slices."""
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        flat = value.detach().reshape(-1)
        for begin in range(0, flat.numel(), 1_000_000):
            result.update(flat[begin:begin+1_000_000].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def shuffled_bank(memory, seed):
    """Destroy addresses while retaining each head's exact learned feature rows."""
    if hasattr(memory, "shuffle_rows"):
        memory.shuffle_rows(seed)
        return
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for table in memory.tables:
        permutation = torch.randperm(table.num_embeddings, generator=generator)
        with torch.no_grad():
            table.weight.copy_(table.weight.detach()[permutation].clone())


def optimizer_for(reader):
    return torch.optim.AdamW(reader.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)


def reader_checkpoint(directory, reader, optimizer, data, step, tokens, identity, settings):
    """Atomic, dependency-bound adapter restart payload; no duplicate base/bank."""
    payload = {"schema_version": 1, "artifact_type": f"{settings.get('memory_kind', 'xmem')}-reader-only",
        "reader": reader.state_dict(), "optimizer": optimizer.state_dict(),
        "data": data, "step": step, "tokens": tokens, "identity": identity,
        "settings": settings, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()}
    destination = directory / "reader-state.pt"
    temporary = directory / "reader-state.pt.partial"
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    atomic_json(directory / "reader-manifest.json", {"artifact_type": payload["artifact_type"],
        "step": step, "new_valid_targets": tokens, "sha256": sha256(destination),
        "bytes": destination.stat().st_size, "identity": identity})


def evaluate(model, injection, token_dir, parent, blocks):
    data = TokenBlocks(token_dir, parent["config"]["sequence_length"], split="validation",
                       seed=parent["config"]["seed"], shards=parent["validation_shards"])
    if data.num_blocks < blocks:
        raise RuntimeError("Frozen validation cannot repeat blocks")
    rows, total, count, hits, positions = [], 0.0, 0, 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for _ in range(blocks):
            index = int(data.order[data.cursor])
            batch = torch.from_numpy(data.next_batch(1))
            x, y = batch[:, :-1], batch[:, 1:]
            if injection.enabled:
                injection.batch(x)
                hits += injection.metrics["active_fraction"] * x.numel()
                positions += x.numel()
            x, y = x.cuda(), y.cuda()
            mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
            nll = float(chunked_loss(model, x, y, parent["config"]["loss_chunk_size"],
                                   loss_mask=mask, reduction="sum"))
            valid = int(mask.sum())
            if not math.isfinite(nll) or valid <= 0:
                raise RuntimeError("Invalid held-out NLL")
            rows.append({"block_indices": [index], "nll": nll, "valid_targets": valid})
            total, count = total + nll, count + valid
    return {"loss": total/count, "nll": total, "valid_targets": count, "blocks": rows,
        "protocol": f"fixed-8k-{blocks}-blocks-document-boundaries-v2", "seed": data.seed,
        "shards": parent["validation_shards"], "inventory_fingerprint": data.fingerprint,
        "memory_hit_fraction": hits/positions if positions else None}


def qualitative(model, injection, tokenizer_path, max_tokens=24):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    suppressed = generation_suppression_ids(json.loads(tokenizer_path.read_text()),
        tokenizer.get_vocab_size(with_added_tokens=True), model.config.vocab_size,
        model.config.eos_token_id, model.config.pad_token_id)
    results = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for prompt in PROMPTS:
            prefix = [model.config.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
            generated = []
            for _ in range(max_tokens):
                x = torch.tensor([prefix + generated])
                if injection.enabled:
                    injection.batch(x)
                hidden = model.model(input_ids=x.cuda(), use_cache=False).last_hidden_state[:, -1]
                logits = model.lm_head(hidden).float()
                logits[:, suppressed] = -torch.inf
                next_id = int(logits.argmax(-1))
                generated.append(next_id)
                if next_id == model.config.eos_token_id:
                    break
            results.append({"prompt": prompt, "continuation": tokenizer.decode(generated,
                skip_special_tokens=False), "tokens": len(generated), "eos": generated[-1] == model.config.eos_token_id})
    return results


def branch(args):
    started, directory = time.monotonic(), args.session_dir / args.branch
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume_branch and any((directory/name).exists() for name in
            ("status.json", "reader-state.pt", "history.json", "metrics.jsonl")):
        raise ValueError("A fresh branch may not overwrite a previous experiment")
    previous_status = json.loads((directory / "status.json").read_text()) if args.resume_branch else {}
    spent = previous_status.get("cumulative_seconds", previous_status.get("total_seconds", 0))
    remaining = args.wall_seconds - spent
    if remaining < 180:
        raise ValueError("The branch's original wall budget is exhausted")
    status = {"phase": "loading", "pid": os.getpid(), "branch": args.branch,
              "step": 0, "new_valid_targets": 0, "started_unix": time.time()}
    for key in ("wandb_id", "wandb_url"):
        if key in previous_status:
            status[key] = previous_status[key]
    def report(**values):
        status.update(values, updated_unix=time.time(), cumulative_seconds=spent+time.monotonic()-started)
        atomic_json(directory / "status.json", status)
    report()
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    tracker = injection = None
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The XMemTransfer pilot requires BF16 CUDA")
        inputs = json.loads((args.session_dir / "inputs.json").read_text())
        if inputs.get("memory_kind", "xmem") != args.memory_kind:
            raise ValueError("Memory backend differs from the admitted session")
        if inputs.get("training_data_source", "parent") != args.training_data_source:
            raise ValueError("Training data source differs from the admitted session")
        if inputs.get("profile", "pilot") != args.profile:
            raise ValueError("Experiment profile differs from the admitted session")
        for path, digest in inputs["implementation"].items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Pilot implementation differs from the bound session")
        identity = {"parent": inputs["parent"], "donor": inputs["donor"], "condition": args.branch,
                    "implementation": inputs["implementation"]}
        parent = torch.load(args.checkpoint / "training_state.pt", weights_only=True,
                            map_location="cpu", mmap=True)
        parent.pop("optimizer")
        if parent["config"]["sequence_length"] != 8192:
            raise ValueError("This pilot preserves the parent's 8K reader")
        for name, digest in inputs["parent"]["files"].items():
            if sha256(args.checkpoint / name) != digest:
                raise RuntimeError(f"Parent publication changed: {name}")
        for name, record in inputs["donor"]["files"].items():
            if sha256(args.donor_dir / name) != record["sha256"]:
                raise RuntimeError(f"Donor publication changed: {name}")
        selected_cursor = select_training_cursor(parent, args.training_data_source)
        if "initial_data_cursor" in inputs and selected_cursor != inputs["initial_data_cursor"]:
            raise RuntimeError("Selected training cursor differs from admission")
        data = GrowingTokenBlocks(args.token_dir, 8192, seed=parent["config"]["seed"])
        data.load_state_dict(selected_cursor)
        if data.known != inputs["chunks"] or data.state_dict() != selected_cursor:
            raise RuntimeError("Pilot must restore the exact frozen parent inventory and data cursor")
        initial_cursor = copy.deepcopy(data.state_dict())
        if sha256(args.token_dir / "manifest.json") != inputs["token_summary_sha256"]:
            raise RuntimeError("Token summary changed since admission")
        model = load_engram_model(args.checkpoint).float().cuda().eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        model.config._attn_implementation = "sdpa"
        if args.memory_kind == "graft":
            from .grafting import GraftMemory, GraftReader
            bank = GraftMemory.from_directory(args.donor_dir, args.checkpoint/"tokenizer.json")
            addressing = bank.addressing
            reader_class = GraftReader
        else:
            bank = FrozenMemory.from_directory(args.donor_dir)
            addressing = WordBoundaryAddressing.from_tokenizer_json(args.checkpoint / "tokenizer.json", bank.config)
            reader_class = TargetReader
        if args.branch == "shuffled":
            shuffled_bank(bank, 20261007)
        bank = bank.cuda()
        torch.manual_seed(42)
        reader = reader_class(model.config.hidden_size, bank.d_mem).float().cuda()
        initial_reader_digest = tensor_digest(reader)
        injection = ReaderInjection(model, reader, bank, addressing, layer=4)
        frozen_parent_digest, frozen_bank_digest = tensor_digest(model), tensor_digest(bank)
        settings = {"max_steps": args.max_steps, "sequence_length": 8192, "micro_batch_size": 1,
            "accumulation": 16, "reader_lr": 1e-3, "warmup_steps": 16, "eval_blocks": args.eval_blocks,
            "injection": "pre-layer4-after-native-engram-layer3", "residual_rms_target": .005,
            "seed": 42, "shuffled_seed": 20261007, "parent_frozen": True, "bank_frozen": True,
            "wall_seconds": args.wall_seconds,
            "donor_dtype": str(next(bank.parameters()).dtype), "reader_parameters": sum(p.numel() for p in reader.parameters()),
            "initial_reader_sha256": initial_reader_digest, "loss_policy": "document-boundaries-v2"}
        if args.memory_kind == "graft":
            settings.update(memory_kind="graft", lookup="exact-longest-suffix-2-3-4",
                native_engram_on_hits=True, bank_trainable=False, teacher_loaded_during_adaptation=False,
                short_conv_kernel=4, short_conv_document_resets=True)
        if args.profile != "pilot":
            settings.update(profile=args.profile, evaluation_steps=sorted(evaluation_steps(args.max_steps, args.profile)))
        if args.training_data_source != "parent":
            settings.update(training_data_source=args.training_data_source,
                training_scope="Local replay of saved retired FineWeb cursor; not production mixture continuation")
        step = tokens = 0
        history = []
        optimizer = optimizer_for(reader)
        resume = directory / "reader-state.pt"
        if args.resume_branch:
            restart_manifest = json.loads((directory/"reader-manifest.json").read_text())
            if resume.stat().st_size != restart_manifest["bytes"] or sha256(resume) != restart_manifest["sha256"]:
                raise RuntimeError("Reader restart checksum mismatch")
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if state["identity"] != identity or state["settings"] != settings:
                raise ValueError("Reader resume dependencies/settings changed")
            reader.load_state_dict(state["reader"])
            optimizer.load_state_dict(state["optimizer"])
            data.load_state_dict(state["data"])
            step, tokens = state["step"], state["tokens"]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            history = [h for h in json.loads((directory / "history.json").read_text()) if h["step"] <= step]
            # The state is saved before evaluation. A restart can therefore be
            # newer than its last published validation; never relabel old NLL.
            if not history or history[-1]["step"] != step:
                result = evaluate(model, injection, args.token_dir, parent, args.eval_blocks)
                result["qualitative"] = qualitative(model, injection, args.checkpoint / "tokenizer.json")
                history.append({"step": step, "new_valid_targets": tokens, "validation": result})
        else:
            sample = torch.from_numpy(data.next_batch(1)[:, :-1])
            injection.batch(sample)
            if args.memory_kind == "graft" and injection.metrics["active_fraction"] < .1:
                raise RuntimeError("Exact graft coverage is below the operational 10% minimum")
            injection.calibration_target = .005
            injection.measure = True
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                model.model(input_ids=sample.cuda(), use_cache=False)
            calibration = dict(injection.metrics)
            injection.calibration_target = None
            injection.measure = False
            data.load_state_dict(initial_cursor)
            if not .004 <= calibration["relative_rms"] <= .006:
                raise RuntimeError("Reader calibration failed")
            atomic_json(directory / "calibration.json", calibration)
            injection.enabled = False
            baseline = evaluate(model, injection, args.token_dir, parent, args.eval_blocks)
            baseline["qualitative"] = qualitative(model, injection, args.checkpoint / "tokenizer.json")
            injection.enabled = True
            initial = evaluate(model, injection, args.token_dir, parent, args.eval_blocks)
            if abs(initial["loss"] - baseline["loss"]) > .02:
                raise RuntimeError("Calibrated reader changed initial validation by more than .02 nats")
            initial["qualitative"] = qualitative(model, injection, args.checkpoint / "tokenizer.json")
            atomic_json(directory / "baseline.json", baseline)
            history.append({"step": 0, "new_valid_targets": 0, "validation": initial})
        atomic_json(directory / "settings.json", settings)
        atomic_json(directory / "history.json", history)
        reader_checkpoint(directory, reader, optimizer, data.state_dict(), step, tokens, identity, settings)
        if not args.no_wandb:
            try:
                import wandb
                tracker = wandb.init(project="MicroLlama2", entity=args.wandb_entity,
                    group=args.session_dir.name, job_type=f"{args.memory_kind}-reader-pilot", name=f"{args.session_dir.name}-{args.branch}",
                    id=status.get("wandb_id") if args.resume_branch else None, resume="allow",
                    dir=str(directory), config={**settings, "parent_step": parent["step"], "parent_tokens": parent["tokens"],
                        "donor_repository": inputs["donor"]["repository"], "donor_revision": inputs["donor"]["revision"]},
                    settings=wandb.Settings(init_timeout=30))
                report(wandb_url=tracker.url, wandb_id=tracker.id)
            except Exception as error:
                report(wandb_error=f"{type(error).__name__}: {error}")
        def track(record):
            if tracker:
                try:
                    tracker.log(record, step=record["step"])
                except Exception as error:
                    report(wandb_error=f"{type(error).__name__}: {error}")
        def track_validation(record):
            values = {"step": record["step"], "validation/loss": record["validation"]["loss"]}
            if record["validation"].get("memory_hit_fraction") is not None:
                values["validation/memory_hit_fraction"] = record["validation"]["memory_hit_fraction"]
            if tracker and record["validation"].get("qualitative"):
                import wandb
                values["quality/fixed_prompts"] = wandb.Table(
                    columns=["step", "condition", "prompt", "continuation", "tokens", "eos"],
                    data=[[record["step"], args.branch, row["prompt"], row["continuation"],
                           row["tokens"], row["eos"]] for row in record["validation"]["qualitative"]])
            track(values)
        if tracker and not args.resume_branch:
            import wandb
            track({"step": 0, "parent/validation_loss": baseline["loss"],
                "quality/parent_fixed_prompts": wandb.Table(
                    columns=["prompt", "continuation", "tokens", "eos"],
                    data=[[row["prompt"], row["continuation"], row["tokens"], row["eos"]]
                          for row in baseline["qualitative"]])})
        track_validation(history[-1])
        torch.cuda.reset_peak_memory_stats()
        report(phase="training", step=step, new_valid_targets=tokens,
               parent_step=parent["step"], parent_tokens=parent["tokens"], settings=settings)
        scheduled_evaluations = evaluation_steps(args.max_steps, args.profile)
        verified_train_chunks = set()
        while step < args.max_steps and not stop and time.monotonic() - started < remaining - 120:
            update_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            nll = valid_total = 0
            first_metrics = {}
            before_update = reader.w_v.weight.detach().clone() if step == 0 else None
            reader.train()
            for micro in range(16):
                cpu_batch = torch.from_numpy(data.next_batch(1, stop_requested=lambda: stop))
                if data.current not in verified_train_chunks:
                    path = args.token_dir / data.current / "train.bin"
                    if sha256(path) != data.metadata[data.current]["splits"]["train"]["sha256"]:
                        raise RuntimeError("Reader-fitting chunk checksum failed")
                    verified_train_chunks.add(data.current)
                if data.known != inputs["chunks"]:
                    raise RuntimeError("Pilot token inventory changed")
                x, y = cpu_batch[:, :-1], cpu_batch[:, 1:]
                injection.batch(x)
                injection.measure = micro == 0
                x, y = x.cuda(), y.cuda()
                mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    loss = chunked_loss(model, x, y, parent["config"]["loss_chunk_size"],
                                        loss_mask=mask, reduction="sum")
                loss.backward()
                if micro == 0:
                    first_metrics = dict(injection.metrics)
                nll += float(loss.detach())
                valid_total += int(mask.sum())
            normalize_accumulated_gradients(reader, valid_total)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in reader.parameters()):
                raise RuntimeError("Missing or nonfinite reader gradients")
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0, error_if_nonfinite=True)
            if float(norm) <= 0:
                raise RuntimeError("Reader did not receive an active gradient")
            rate = 1e-3 * min(1.0, (step + 1) / 16)
            optimizer.param_groups[0]["lr"] = rate
            optimizer.step()
            step, tokens = step + 1, tokens + valid_total
            torch.cuda.synchronize()
            if step == 1:
                if torch.equal(reader.w_v.weight.detach(), before_update):
                    raise RuntimeError("Reader parameters did not update")
                if any(p.grad is not None or p.requires_grad for p in model.parameters()):
                    raise RuntimeError("Frozen parent received gradients")
                report(qualification_passed=True)
            record = {"step": step, "new_valid_targets": tokens, "train/loss": nll / valid_total,
                "train/gradient_norm": float(norm), "train/learning_rate": rate,
                "performance/tokens_per_second": valid_total/(time.monotonic()-update_start),
                "performance/peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
            record.update({"memory/" + key: value for key, value in first_metrics.items()})
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            track(record)
            report(step=step, new_valid_targets=tokens, last_metrics=record)
            if step in scheduled_evaluations:
                reader_checkpoint(directory, reader, optimizer, data.state_dict(), step, tokens, identity, settings)
                injection.measure = False
                result = evaluate(model, injection, args.token_dir, parent, args.eval_blocks)
                result["qualitative"] = qualitative(model, injection, args.checkpoint / "tokenizer.json")
                history.append({"step": step, "new_valid_targets": tokens, "validation": result})
                atomic_json(directory / "history.json", history)
                track_validation(history[-1])
        # Save progress first even if final held-out checks consume the reserve.
        reader_checkpoint(directory, reader, optimizer, data.state_dict(), step, tokens, identity, settings)
        if history[-1]["step"] != step:
            result = evaluate(model, injection, args.token_dir, parent, args.eval_blocks)
            result["qualitative"] = qualitative(model, injection, args.checkpoint / "tokenizer.json")
            history.append({"step": step, "new_valid_targets": tokens, "validation": result})
            atomic_json(directory / "history.json", history)
            track_validation(history[-1])
        if tensor_digest(model) != frozen_parent_digest or tensor_digest(bank) != frozen_bank_digest:
            raise RuntimeError("Frozen base or donor weights changed")
        report(phase="completed" if step == args.max_steps else "stopped", step=step,
               new_valid_targets=tokens, frozen_parent_verified=True, frozen_bank_verified=True,
               total_seconds=time.monotonic()-started, final_validation_loss=history[-1]["validation"]["loss"])
    except Exception as error:
        report(phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if injection:
            injection.close()
        if tracker:
            tracker.finish(exit_code=0 if status["phase"] in ("completed", "stopped") else 1)


def summarize(session):
    from scripts.benchmarks.run_optimizer_memory_screen import paired_block_comparison
    statuses, histories = {}, {}
    for name in CONDITIONS:
        p = session / name / "status.json"
        if p.exists():
            statuses[name] = json.loads(p.read_text())
        p = session / name / "history.json"
        if p.exists():
            histories[name] = json.loads(p.read_text())
    inputs = json.loads((session / "inputs.json").read_text())
    result = {"conditions": statuses, "parent": inputs["parent"],
              "memory_kind": inputs.get("memory_kind", "xmem"),
              "profile": inputs.get("profile", "pilot"),
              "training_data_source": inputs.get("training_data_source", "parent"),
              "scope": "Frozen-parent reader adaptation; not a new full pretraining run or evidence of faster pretraining",
              "automatic_promotion": False}
    if len(histories) == 2:
        left = {(r["step"], r["new_valid_targets"]): r["validation"] for r in histories["shuffled"]}
        right = {(r["step"], r["new_valid_targets"]): r["validation"] for r in histories["pretrained"]}
        common = sorted(set(left) & set(right))
        result["matched_history"] = [{"step": k[0], "new_valid_targets": k[1],
            **paired_block_comparison(left[k], right[k])} for k in common]
        result["parent_comparisons"] = {}
        for name in CONDITIONS:
            baseline = json.loads((session/name/"baseline.json").read_text())
            result["parent_comparisons"][name] = paired_block_comparison(baseline, histories[name][-1]["validation"])
    atomic_json(session / "summary.json", result)
    return result


def wait_for_bank(args):
    """Bounded dependency wait outside the recipient training wall budgets.

    The offline builder publishes its final directory atomically. Do not hold
    its GPU lock while waiting, or admit a bank before its builder releases CUDA.
    The coordinator still verifies all checksums and acquires its own lock.
    """
    started = time.monotonic()
    while time.monotonic() - started < args.wait_for_bank_seconds:
        ready = (args.donor_dir / "source.json").is_file()
        worker_path = args.donor_dir.with_name(args.donor_dir.name + ".worker-status.json")
        if not ready and worker_path.is_file():
            worker = json.loads(worker_path.read_text())
            if worker.get("phase") in ("launching", "running") and worker.get("pid"):
                try:
                    os.kill(worker["pid"], 0)
                except ProcessLookupError:
                    worker = {**worker,"phase":"failed","error":"Builder exited without publishing a bank"}
            if worker.get("phase") == "failed":
                atomic_json(args.session_dir / "preparation-status.json",
                    {"phase":"failed","error":"Offline bank builder failed","builder":worker,"pid":os.getpid()})
                raise RuntimeError(f"Offline bank builder failed: {worker.get('error', 'unknown error')}")
        if ready:
            try:
                with gpu_lock():
                    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                        "--format=csv,noheader"], text=True, timeout=10)
                    if not any(line.strip() != str(os.getpid()) for line in apps.splitlines() if line.strip()):
                        atomic_json(args.session_dir / "preparation-status.json",
                            {"phase": "bank-ready", "pid": os.getpid(), "wait_seconds": time.monotonic()-started})
                        return
            except BlockingIOError:
                pass
        atomic_json(args.session_dir / "preparation-status.json", {
            "phase": "waiting-for-bank" if not ready else "waiting-for-builder-release",
            "pid": os.getpid(), "bank": str(args.donor_dir),
            "elapsed_seconds": time.monotonic()-started, "limit_seconds": args.wait_for_bank_seconds})
        time.sleep(5)
    atomic_json(args.session_dir / "preparation-status.json",
        {"phase": "failed", "error": "Bank preparation wait exceeded its original budget", "pid": os.getpid()})
    raise RuntimeError("Bank preparation did not finish and release CUDA within its wait budget")


def coordinator(args):
    session = args.session_dir
    session.mkdir(parents=True, exist_ok=True)
    if (session / "driver-status.json").exists():
        raise ValueError("Choose a fresh session; branch-only resumes do not reset pilot budgets")
    if args.wait_for_bank_seconds:
        if (session / "preparation-status.json").exists():
            raise ValueError("Choose a fresh session; preparation waits do not reset their budgets")
        wait_for_bank(args)
    with gpu_lock() as lock_fd:
        # Only adapters are written; a separate small-artifact budget preserves
        # 20GiB even on a host that cannot fit another full 4.66GB checkpoint.
        artifact_budget = (512 if args.memory_kind == "graft" else 256) * 2**20
        if shutil.disk_usage(session).free < 20*2**30 + artifact_budget:
            raise RuntimeError("Reader-only pilot needs its artifact budget plus 20GiB untouched disk reserve")
        idle_gpu()
        identity = input_identity(args.checkpoint, args.token_dir)
        manifest = json.loads((args.checkpoint / "checkpoint_manifest.json").read_text())
        for name, record in manifest["files"].items():
            if (args.checkpoint / name).stat().st_size != record["bytes"] or sha256(args.checkpoint/name) != record["sha256"]:
                raise RuntimeError(f"Parent checksum mismatch: {name}")
        source = json.loads((args.donor_dir / "source.json").read_text())
        if args.memory_kind == "graft":
            if source.get("kind") != "memory-grafting-additive-v1":
                raise ValueError("Wrong graft-bank publication")
            donor_files = {name:source["files"][name] for name in ("graft-bank.pt", "graft-config.json")}
        else:
            donor_files = {"memory_best.pt":source["files"]["memory_best.pt"],
                           "memory_config.json":{"sha256":sha256(args.donor_dir/"memory_config.json")}}
        for name, record in donor_files.items():
            if sha256(args.donor_dir/name) != record["sha256"]:
                raise RuntimeError(f"Bank checksum mismatch: {name}")
        implementation_files = [Path(__file__),Path(__file__).with_name("xmem.py"),Path(__file__).with_name("gpu.py")]
        if args.memory_kind == "graft":
            implementation_files.append(Path(__file__).with_name("grafting.py"))
        parent_state = torch.load(args.checkpoint/"training_state.pt", map_location="cpu", weights_only=True, mmap=True)
        if (parent_state["step"], parent_state["tokens"]) != (manifest["step"], manifest["tokens"]):
            raise ValueError("Parent state step/tokens differ from its checkpoint manifest")
        selected_cursor = select_training_cursor(parent_state, args.training_data_source)
        data = GrowingTokenBlocks(args.token_dir, parent_state["config"]["sequence_length"], seed=parent_state["config"]["seed"])
        data.load_state_dict(selected_cursor)
        if data.known != identity["chunks"] or data.state_dict() != selected_cursor:
            raise RuntimeError("Admission requires the exact selected cursor and immutable local chunk inventory")
        validation_checksums = verify_validation_inventory(args.token_dir, parent_state["validation_shards"], identity)
        del parent_state
        inputs = {"memory_kind":args.memory_kind, "profile":args.profile,
            "training_data_source":args.training_data_source, "initial_data_cursor":selected_cursor,
            "parent": {"path": identity["checkpoint"], "step": manifest["step"], "tokens": manifest["tokens"],
                    "manifest_sha256": sha256(args.checkpoint/"checkpoint_manifest.json"), "files": identity["files"]},
            "chunks": identity["chunks"], "token_summary_sha256": identity["summary_sha256"],
            "validation_binary_checksums": validation_checksums,
            "implementation": {str(path.resolve()):sha256(path) for path in implementation_files},
            "donor": {"repository": source["repo"], "revision": source["revision"], "files": donor_files},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        atomic_json(session/"inputs.json", inputs)
        status = {"phase": "running", "pid": os.getpid(), "started_unix": time.time(),
            "memory_kind":args.memory_kind, "profile":args.profile,
            "max_steps_per_condition": args.max_steps, "wall_seconds_per_condition": args.wall_seconds,
            "conditions": {}, "parent": inputs["parent"]["path"]}
        stopped, child = False, None
        def request_stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        matched_steps = args.max_steps
        for name in CONDITIONS:
            if stopped:
                break
            status.update(active_branch=name, updated_unix=time.time())
            atomic_json(session/"driver-status.json", status)
            command = [sys.executable, "-m", __package__ + ".xmem_pilot", "--branch", name,
                "--checkpoint", str(args.checkpoint), "--token-dir", str(args.token_dir),
                "--donor-dir", str(args.donor_dir), "--session-dir", str(session),
                "--max-steps", str(matched_steps), "--eval-blocks", str(args.eval_blocks),
                "--wall-seconds", str(args.wall_seconds), "--wandb-entity", args.wandb_entity,
                "--memory-kind", args.memory_kind, "--profile", args.profile,
                "--training-data-source", args.training_data_source,
                "--gpu-lock-fd", str(lock_fd)]
            if args.no_wandb:
                command.append("--no-wandb")
            begin = time.monotonic()
            with (session/f"{name}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock_fd,))
                status.update(active_pid=child.pid)
                atomic_json(session/"driver-status.json", status)
                while child.poll() is None:
                    if time.monotonic()-begin > args.wall_seconds+60:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(2)
                code = child.wait()
            status["conditions"][name] = {"exit_code": code, "seconds": time.monotonic()-begin}
            status["active_pid"] = None
            atomic_json(session/"driver-status.json", status)
            summarize(session)
            if code != 0:
                break
            actual = json.loads((session/name/"status.json").read_text())
            matched_steps = actual["step"]
            if matched_steps == 0:
                break
        status.update(phase="stopped" if stopped else "completed" if len(status["conditions"]) == 2 and
            all(r["exit_code"] == 0 for r in status["conditions"].values()) else "failed",
            active_pid=None, active_branch=None, ended_unix=time.time())
        atomic_json(session/"driver-status.json", status)
        if status["phase"] == "completed" and matched_steps < args.max_steps:
            status["phase"] = "budget_stopped"
            atomic_json(session/"driver-status.json", status)
        summarize(session)
        if status["phase"] == "failed":
            raise RuntimeError("A pilot condition failed; inspect its status and log")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--donor-dir", type=Path, required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--branch", choices=CONDITIONS)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--eval-blocks", type=int, default=128)
    p.add_argument("--wall-seconds", type=float, default=2400)
    p.add_argument("--resume-branch", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "wangxiaotian2007"))
    p.add_argument("--gpu-lock-fd", type=int, help=argparse.SUPPRESS)
    p.add_argument("--memory-kind", choices=("xmem", "graft"), default="xmem")
    p.add_argument("--profile", choices=("pilot", "graft-confirmation"), default="pilot",
        help="Explicit longer graft comparison: at most 2048 updates and four hours per condition; fresh session required")
    p.add_argument("--training-data-source", choices=("parent", "retired-fineweb"), default="parent",
        help="Explicit local replay of a mixture checkpoint's saved retired FineWeb cursor; never resets production data")
    p.add_argument("--wait-for-bank-seconds", type=float, default=0,
        help="For a fresh graft coordinator: bounded wait for an offline bank and idle GPU before admission")
    args = p.parse_args()
    try:
        limit, wall_limit = pilot_limits(args.memory_kind, args.profile)
    except ValueError as error:
        p.error(str(error))
    if not 1 <= args.max_steps <= limit or not 1 <= args.eval_blocks <= 128 or not 180 <= args.wall_seconds <= wall_limit:
        p.error(f"Pilot bounds: 1–{limit} updates, 1–128 eval blocks, 180–{wall_limit} seconds per condition")
    if args.training_data_source != "parent" and args.profile != "graft-confirmation":
        p.error("Retired FineWeb replay is only supported by an explicit graft-confirmation profile")
    if args.resume_branch and not args.branch:
        p.error("A resume must name one condition; it does not relaunch the whole experiment")
    if not 0 <= args.wait_for_bank_seconds <= 3600 or (args.wait_for_bank_seconds and
            (args.memory_kind != "graft" or args.branch or args.resume_branch)):
        p.error("Bank waits are limited to one hour and require a fresh --memory-kind graft coordinator")
    for name in ("checkpoint", "token_dir", "donor_dir", "session_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.branch:
        with gpu_lock(args.gpu_lock_fd):
            if args.gpu_lock_fd is None:
                idle_gpu()
            branch(args)
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
