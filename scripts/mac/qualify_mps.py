"""Small, bounded CPU/MPS compatibility gates for the existing Engram model.

This uses synthetic tokens and a tiny model. It does not download data or
weights, continue the production run, or claim production-context qualification.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microllama.training.engram import (
    EngramMemory, EngramQwen3Config, EngramState, LOOKUP_V2,
    build_engram_model,
)
from microllama.training.objective import (
    chunked_loss, normalize_accumulated_gradients, target_mask,
)
from microllama.training.optimizers import build_optimizer


TINY = {
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6,
    "rope_theta": 10000, "max_position_embeddings": 64,
    "tie_word_embeddings": True, "attention_dropout": 0.0,
    "pad_token_id": 0, "eos_token_id": 7, "bos_token_id": 1,
}
DEFAULT_DIRECTORY = Path("/Volumes/TrainData/MicroLlamaV2/mac-continuation-50b/qualification")


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def close_metrics(actual, expected, *, atol, rtol):
    """Refuse nonfinite or out-of-tolerance values and retain useful errors."""
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Nonfinite parity tensor")
    error = (actual - expected).abs()
    maximum = float(error.max()) if error.numel() else 0.0
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {"max_absolute_error": maximum, "atol": atol, "rtol": rtol}


def hash_fixture():
    # Large IDs catch truncation even if a backend accepts int64 tensors.
    mapping = list(range(65536))
    mapping[13] = 12
    config = EngramQwen3Config(
        **{**TINY, "vocab_size": 65536}, engram_table_size=97,
        engram_embedding_dim=8, engram_layers=(0, 1),
        engram_lookup_version=LOOKUP_V2, engram_hash_heads=4,
        engram_token_id_map=mapping,
    )
    memory = EngramMemory(config).float()
    ids = torch.tensor([
        [2, 255, 256, 32767, 32768, 65535, 7, 13, 14, 15, 0, 12, 14, 15, 7, 0],
        [13, 12, 13, 12, 65535, 32768, 0, 2, 3, 4, 7, 5, 6, 8, 9, 10],
    ], dtype=torch.long)
    history = EngramState(torch.tensor([[-1, 12], [65535, 32768]], dtype=torch.long))
    return memory, ids, history


def assert_hash_parity(reference, actual, ids, history):
    expected = reference._indices(ids, history)
    observed = actual._indices(ids.to("mps"), history.to("mps"))
    labels = ("bigram_indices", "trigram_indices", "bigram_valid", "trigram_valid")
    checks = {}
    for label, left, right in zip(labels, expected, observed):
        right = right.detach().cpu()
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"Exact integer/boundary parity failed: {label}")
        checks[label] = {"shape": list(left.shape), "dtype": str(left.dtype), "sha256": digest(left)}
    left = reference._next_state(ids, history).history_ids
    right = actual._next_state(ids.to("mps"), history.to("mps")).history_ids.detach().cpu()
    if not torch.equal(left, right):
        raise AssertionError("Exact next suffix-state parity failed")
    checks["history"] = {"values": left.tolist(), "sha256": digest(left)}
    return checks


def native_hash_check():
    reference, ids, history = hash_fixture()
    actual = copy.deepcopy(reference).to("mps")
    return assert_hash_parity(reference, actual, ids, history)


def adapter_hash_check():
    from microllama.training.mac import install_mps_engram_hash_adapter
    reference, ids, history = hash_fixture()
    actual = copy.deepcopy(reference).to("mps")
    # The installer accepts a complete model, so use its actual memory nesting.
    holder = torch.nn.Module()
    holder.model = torch.nn.Module()
    holder.model.engram = actual
    install_mps_engram_hash_adapter(holder)
    result = assert_hash_parity(reference, actual, ids, history)
    # Canonical aliases must retain identical lookups after the explicit adapter.
    alias_ids = ids.clone()
    alias_ids[alias_ids == 13] = 12
    alias = actual._indices(alias_ids.to("mps"), history.to("mps"))
    normal = actual._indices(ids.to("mps"), history.to("mps"))
    for left, right in zip(alias, normal):
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError("Canonical-token alias semantics changed")
    result["adapter"] = "explicit CPU int64 hashing and boundary state, MPS embeddings"
    return result


def sdpa_check():
    generator = torch.Generator(device="cpu").manual_seed(20261018)
    source = [torch.randn(2, 4, 17, 8, generator=generator) for _ in range(3)]
    cpu = [x.clone().requires_grad_(True) for x in source]
    mps = [x.to("mps").requires_grad_(True) for x in source]
    expected = F.scaled_dot_product_attention(*cpu, is_causal=True, dropout_p=0.0)
    actual = F.scaled_dot_product_attention(*mps, is_causal=True, dropout_p=0.0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    checks = {"output": close_metrics(actual, expected, atol=1e-5, rtol=1e-4)}
    for label, left, right in zip(("q", "k", "v"), cpu, mps):
        checks[f"gradient_{label}"] = close_metrics(right.grad, left.grad, atol=1e-5, rtol=1e-4)
    return {"shape": [2, 4, 17, 8], "dtype": "float32", "causal": True,
            "dropout": 0.0, "checks": checks}


def tiny_model():
    mapping = list(range(TINY["vocab_size"]))
    mapping[13] = 12
    model = build_engram_model(
        TINY, table_size=97, embedding_dim=8, layers=(0, 1),
        residual_scale=1.0, lookup_version=LOOKUP_V2, hash_heads=4,
        token_id_map=mapping,
    ).float()
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    return model.train()


def padding_embedding_check(padding_backend):
    from microllama.training.mac import install_mps_padding_adapter
    torch.manual_seed(43)
    cpu = torch.nn.Embedding(8, 4, padding_idx=0)
    # The trained tied PAD row need not be zero: preserve its forward value.
    with torch.no_grad():
        cpu.weight[0].copy_(torch.tensor([.25, -.5, .75, 1.0]))
    mps = copy.deepcopy(cpu).to("mps")
    if padding_backend == "corrected":
        class Holder:
            def get_input_embeddings(self):
                return mps
        install_mps_padding_adapter(Holder())
    ids = torch.tensor([[0, 1, 0, 2, 3]])
    reference, actual = cpu(ids), mps(ids.to("mps"))
    forward = close_metrics(actual, reference, atol=0, rtol=0)
    # Exercise input lookup and tied output-head contribution separately.
    hidden = torch.tensor([[.2, .3, -.1, .4], [.1, -.2, .5, .3]])
    for embedding, value, device in ((cpu, reference, "cpu"), (mps, actual, "mps")):
        value.square().sum().backward()
        lookup_pad_grad = embedding.weight.grad[0].detach().cpu().clone()
        if padding_backend == "corrected" and not torch.equal(lookup_pad_grad, torch.zeros_like(lookup_pad_grad)):
            raise AssertionError("PAD input lookup gradient was not suppressed")
        F.linear(hidden.to(device), embedding.weight).square().sum().backward()
    gradient = close_metrics(mps.weight.grad, cpu.weight.grad, atol=1e-6, rtol=1e-5)
    if not bool(cpu.weight.grad[0].abs().sum() > 0):
        raise AssertionError("Tied output-head PAD gradient was incorrectly removed")
    return {"padding_backend": padding_backend, "forward": forward, "gradient": gradient,
            "cpu_tied_pad_gradient": cpu.weight.grad[0].tolist(),
            "mps_tied_pad_gradient": mps.weight.grad[0].cpu().tolist()}


def tiny_gradient_update_check(hash_backend, padding_backend, attention_backend='native'):
    torch.manual_seed(20261018)
    reference = tiny_model()
    actual = copy.deepcopy(reference).to("mps")
    if attention_backend == 'tiled':
        from microllama.training.mac_attention import install_tiled_attention
        install_tiled_attention(actual, query_tile=5)
    if hash_backend == "cpu":
        from microllama.training.mac import install_mps_engram_hash_adapter
        install_mps_engram_hash_adapter(actual)
    if padding_backend == "corrected":
        from microllama.training.mac import install_mps_padding_adapter, install_mps_embedding_reduction_adapter
        install_mps_padding_adapter(actual)
        install_mps_embedding_reduction_adapter(actual)
    rows = torch.tensor([
        [2, 3, 13, 14, 7, 5, 6, 8, 9, 10, 0, 0, 11, 12, 14, 15, 16],
        [2, 7, 4, 7, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ], dtype=torch.long)
    cpu_optimizer = build_optimizer(reference, fused=False, memory_lr_multiplier=5)
    mps_optimizer = build_optimizer(actual, fused=False, memory_lr_multiplier=5)
    # Explicitly avoid a backend choosing a different multi-tensor optimizer.
    for optimizer in (cpu_optimizer, mps_optimizer):
        for group in optimizer.param_groups:
            group.update(foreach=False, capturable=False)
    checks = {}
    losses, counts = [], []
    for device, model in (("cpu", reference), ("mps", actual)):
        values = rows.to(device)
        x, y = values[:, :-1], values[:, 1:]
        mask = target_mask(x, y, 7, 0)
        loss = chunked_loss(model, x, y, chunk_size=7, loss_mask=mask, reduction="sum")
        loss.backward()
        # FP64 totals live on CPU; never allocate an MPS float64 tensor.
        total_nll = float(loss.detach().cpu().double())
        valid = int(mask.sum().detach().cpu())
        normalize_accumulated_gradients(model, valid)
        losses.append(total_nll)
        counts.append(valid)
    if counts[0] != counts[1] or counts[0] < 1:
        raise AssertionError("Boundary-masked valid-target counts differ")
    checks["summed_nll"] = close_metrics(torch.tensor(losses[1]), torch.tensor(losses[0]), atol=2e-4, rtol=2e-5)
    cpu_parameters, mps_parameters = dict(reference.named_parameters()), dict(actual.named_parameters())
    active_memory = []
    gradient_max = 0.0
    for name, expected in cpu_parameters.items():
        observed = mps_parameters[name]
        if expected.grad is None or observed.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        try:
            result = close_metrics(observed.grad, expected.grad, atol=2e-5, rtol=5e-4)
        except AssertionError as error:
            raise AssertionError(f"Gradient mismatch for {name}: {error}") from error
        gradient_max = max(gradient_max, result["max_absolute_error"])
        if ".engram." in name:
            if float(observed.grad.abs().sum().cpu()) == 0:
                raise AssertionError(f"Inactive Engram gradient: {name}")
            active_memory.append(name)
    before = {name: p.detach().cpu().clone() for name, p in actual.named_parameters()}
    cpu_optimizer.step()
    mps_optimizer.step()
    update_max = 0.0
    changed = []
    for name, expected in cpu_parameters.items():
        observed = mps_parameters[name]
        result = close_metrics(observed, expected, atol=2e-5, rtol=5e-4)
        update_max = max(update_max, result["max_absolute_error"])
        if ".engram." in name and not torch.equal(observed.detach().cpu(), before[name]):
            changed.append(name)
    if set(changed) != set(active_memory):
        raise AssertionError("An active Engram parameter did not update")
    if reference.lm_head.weight is not reference.model.embed_tokens.weight or actual.lm_head.weight is not actual.model.embed_tokens.weight:
        raise AssertionError("Tied word embeddings were lost")
    checks.update(max_gradient_absolute_error=gradient_max, max_parameter_absolute_error_after_update=update_max)
    return {"valid_targets": counts[0], "input_positions": int(rows.shape[0] * (rows.shape[1] - 1)),
            "cpu_summed_nll": losses[0], "mps_summed_nll": losses[1],
            "active_updated_memory_parameters": active_memory,
            "checks": checks, "optimizer": "AdamW FP32, foreach=False, fused=False, memory LR multiplier=5",
            "hash_backend": hash_backend, "padding_backend": padding_backend,
            "production_checkpoint_loaded": False}


def tiny_restart_check(hash_backend, padding_backend, directory, attention_backend='native'):
    from microllama.training.mac import (
        cpu_tree, install_mps_engram_hash_adapter, install_mps_padding_adapter,
        portable_adam_state, update,
    )
    def adapt(model):
        if attention_backend == 'tiled':
            from microllama.training.mac_attention import install_tiled_attention
            install_tiled_attention(model, query_tile=5)
        if hash_backend == "cpu":
            install_mps_engram_hash_adapter(model)
        if padding_backend == "corrected":
            install_mps_padding_adapter(model)
            from microllama.training.mac import install_mps_embedding_reduction_adapter
            install_mps_embedding_reduction_adapter(model)
        return model.to("mps")
    def optimizer_for(model):
        optimizer = build_optimizer(model, fused=False, memory_lr_multiplier=5)
        for group in optimizer.param_groups:
            group.update(foreach=False, capturable=False)
        return optimizer
    def rows_for(generator):
        rows = torch.randint(2, 64, (2, 17), generator=generator)
        rows[0, 3], rows[0, 9], rows[1, 7] = 7, 0, 7
        return rows
    def batches_for(rows):
        return [(row[None, :-1].to("mps"), row[None, 1:].to("mps")) for row in rows]
    torch.manual_seed(20261018)
    torch.mps.manual_seed(20261018)
    generator = torch.Generator(device="cpu").manual_seed(51)
    model = adapt(tiny_model())
    optimizer = optimizer_for(model)
    first = rows_for(generator)
    first_update = update(model, optimizer, batches_for(first), chunk_size=7)
    saved_optimizer = cpu_tree(optimizer.state_dict())
    # Simulate the real CUDA checkpoint's backend flags without changing moments.
    for group in saved_optimizer["param_groups"]:
        group["fused"] = True
    saved = {"model": cpu_tree(model.state_dict()), "optimizer": saved_optimizer,
             "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
             "reader_rng": generator.get_state(), "reader_rows_consumed": 2}
    with tempfile.TemporaryDirectory(prefix="tiny-restart-", dir=directory) as temporary:
        checkpoint = Path(temporary) / "state.pt"
        torch.save(saved, checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        second = rows_for(generator)
        expected_random = torch.rand(7, device="mps").cpu()
        second_update = update(model, optimizer, batches_for(second), chunk_size=7)
        expected_weights = cpu_tree(model.state_dict())
        expected_optimizer = cpu_tree(optimizer.state_dict())
        # Serialize the expectation too, preventing CPU scalar-step aliasing.
        expected_file = Path(temporary) / "expected.pt"
        torch.save({"model": expected_weights, "optimizer": expected_optimizer}, expected_file)
        restored_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = tiny_model()
        restored.load_state_dict(restored_state["model"], strict=True)
        restored = adapt(restored)
        restored_optimizer = optimizer_for(restored)
        restored_optimizer.load_state_dict(portable_adam_state(restored_state["optimizer"]))
        if any(group.get("fused") or group.get("foreach") or group.get("capturable")
               for group in restored_optimizer.param_groups):
            raise AssertionError("CUDA optimizer backend flags survived portable restoration")
        if not all(slot["step"].device.type == "cpu" for slot in restored_optimizer.state.values()):
            raise AssertionError("Unfused Adam scalar counters must remain on CPU")
        torch.set_rng_state(restored_state["torch_rng"])
        torch.mps.set_rng_state(restored_state["mps_rng"])
        generator.set_state(restored_state["reader_rng"])
        replay = rows_for(generator)
        if not torch.equal(replay, second):
            raise AssertionError("Restored synthetic data cursor selected different next rows")
        actual_random = torch.rand(7, device="mps").cpu()
        if not torch.equal(actual_random, expected_random):
            raise AssertionError("MPS RNG restoration changed the next random draw")
        actual_update = update(restored, restored_optimizer, batches_for(replay), chunk_size=7)
        expected = torch.load(expected_file, map_location="cpu", weights_only=True)
        maximum = 0.0
        for name, tensor in restored.state_dict().items():
            target = expected["model"][name]
            if tensor.is_floating_point():
                value = close_metrics(tensor, target, atol=1e-7, rtol=1e-6)
                maximum = max(maximum, value["max_absolute_error"])
            elif not torch.equal(tensor.cpu(), target):
                raise AssertionError(f"Restored model buffer changed: {name}")
        observed_optimizer = cpu_tree(restored_optimizer.state_dict())
        if observed_optimizer["param_groups"] != expected["optimizer"]["param_groups"]:
            raise AssertionError("Optimizer group metadata changed after restoration")
        for key, slot in observed_optimizer["state"].items():
            for field, tensor in slot.items():
                close_metrics(tensor, expected["optimizer"]["state"][key][field], atol=1e-7, rtol=1e-6)
    return {"first_update": first_update, "uninterrupted_second_update": second_update,
            "restored_second_update": actual_update, "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_sha256, "next_rows_sha256": digest(second),
            "mps_rng_next_draw_exact": True, "saved_fused_flag_removed": True,
            "max_parameter_absolute_error": maximum,
            "scope": "Tiny synthetic fixture; not a production MixtureTokenBlocks cursor audit"}


def capture(name, function):
    started = time.monotonic()
    try:
        result = function()
        torch.mps.synchronize()
        return {"name": name, "passed": True, "seconds": time.monotonic() - started, "result": result}
    except Exception as error:
        return {"name": name, "passed": False, "seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}


def run(phase, hash_backend, padding_backend, directory, attention_backend='native'):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise RuntimeError("Blanket MPS fallback must be disabled for qualification")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") in ("0", "0.0"):
        raise RuntimeError("Qualification refuses a disabled MPS allocator safety limit")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run on the authorized Mac GPU outside the sandbox")
    torch.set_num_threads(2)
    result = {
        "schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase, "hash_backend_requested": hash_backend, "padding_backend_requested": padding_backend,
        "attention_backend": attention_backend,
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                        "mps_available": True, "blanket_mps_fallback": False},
        "limitations": ["Synthetic fixtures and tiny model only", "No production 8K memory/throughput claim",
                        "No production checkpoint or dataset loaded", "No production training continuation"],
    }
    native = capture("native_int64_hash", native_hash_check)
    result["native_hash_diagnostic"] = native
    if phase == "native-hash":
        result["passed"] = native["passed"]
        return result
    gates = []
    if hash_backend == "cpu":
        gates.append(capture("explicit_cpu_hash_adapter", adapter_hash_check))
    else:
        gates.append(native)
    gates.append(capture("sdpa_forward_backward", sdpa_check))
    gates.append(capture("padding_embedding_forward_backward", lambda: padding_embedding_check(padding_backend)))
    if gates[0]["passed"]:
        gates.append(capture("tiny_engram_gradient_update", lambda: tiny_gradient_update_check(hash_backend, padding_backend, attention_backend)))
    else:
        gates.append({"name": "tiny_engram_gradient_update", "passed": False,
                      "not_run": "Hash gate failed; no model update attempted"})
    if all(item["passed"] for item in gates):
        gates.append(capture("tiny_mps_checkpoint_restart", lambda: tiny_restart_check(hash_backend, padding_backend, directory, attention_backend)))
    result["gates"] = gates
    result["passed"] = all(item["passed"] for item in gates)
    result["native_hash_supported"] = native["passed"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native-hash", "all"), default="all")
    parser.add_argument("--hash-backend", choices=("native", "cpu"), default="native")
    parser.add_argument("--padding-backend", choices=("native", "corrected"), default="native")
    parser.add_argument('--attention-backend', choices=('native', 'tiled'), default='native')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if DEFAULT_DIRECTORY not in output.parents:
        parser.error(f"Output must be inside {DEFAULT_DIRECTORY}")
    if output.exists():
        parser.error("Choose a new receipt path; qualification evidence is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.phase, args.hash_backend, args.padding_backend, output.parent, args.attention_backend)
    except Exception as error:
        result = {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(), "passed": False,
                  "phase": args.phase, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "phase": args.phase, "receipt": str(output),
                      "native_hash_supported": result.get("native_hash_supported")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
