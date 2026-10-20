"""Hardware checks and explicitly unmeasured economic forecasts; no standalone CLI.

Profiles keep the training recipe in train-8k-local.json and change only batch
partitioning/environment. Throughput bounds are engineering priors for the
current Engram 8K recipe, including periodic validation/save time. They are not
confidence intervals or GPU benchmarks. See docs/gpu-economics.md.
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def profiles():
    return json.loads((REPO / "configs/cloud-hardware.json").read_text())


def estimate(target_tokens, already_trained=0, hourly_prices=None, measured_rates=None,
             single_hourly_price=None, single_measured_rate=None):
    """Whole-instance compute cost; measured rates must include monitoring overhead."""
    if target_tokens < 1 or not 0 <= already_trained < target_tokens:
        raise ValueError("Require a positive target and 0 <= already-trained tokens < target")
    registry = profiles()
    rental_profiles = {key: value for key, value in registry.items() if value["gpus"] == 4}
    for values in (hourly_prices, measured_rates):
        if values is not None and (len(values) != len(rental_profiles) or
                any(not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError("Provide three finite positive values, ordered 4090-24, 4090-48, 5090-32")
    for value in (single_hourly_price, single_measured_rate):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Single-GPU price and measured effective rate must be finite and positive")
    remaining = target_tokens - already_trained

    def row(key, profile, price, measured):
        low, central, high = profile["effective_tokens_per_second"]
        if measured is not None:
            low = central = high = measured
        hours = remaining / central / 3600
        return {"profile": key, "label": profile["label"],
                     "basis": "user-supplied measured effective throughput" if measured else "unmeasured planning estimate",
                     "effective_valid_tokens_per_second": central,
                     "hours": hours, "hours_range": [remaining / high / 3600, remaining / low / 3600],
                     "hours_per_billion_tokens": 1e9 / central / 3600,
                     "whole_instance_hourly_price": price,
                     "compute_cost": None if price is None else hours * price,
                     "compute_cost_range": None if price is None else
                        [remaining / high / 3600 * price, remaining / low / 3600 * price]}
    rows = [row(key, profile, hourly_prices[index] if hourly_prices is not None else None,
                measured_rates[index] if measured_rates is not None else None)
            for index, (key, profile) in enumerate(rental_profiles.items())]
    single = row("4090-24-single", registry["4090-24-single"], single_hourly_price, single_measured_rate)
    if single_measured_rate is None:
        single["basis"] = "Measured 20955 update-only targets/s; effective rate includes estimated monitoring/save overhead"
    baseline = rows[0]["effective_valid_tokens_per_second"]
    for row in rows:
        row["break_even_hourly_price_ratio_to_4090_24"] = row["effective_valid_tokens_per_second"] / baseline
    return {"target_tokens": target_tokens, "already_trained_tokens": already_trained,
            "remaining_tokens": remaining, "context_length": 8192,
            "assumptions": "Current ~388M Engram, BF16 compute/FP32 Adam, global batch 131072; periodic validation/save included; setup, transfers, outages, storage and bandwidth charges excluded. Dense may be faster; 16K/32K curriculum is not priced.",
            "current_single_4090": single,
            "profiles": rows}


def cuda_preflight(gpus, profile_key=None):
    """Run on the selected environment, before data preparation or torchrun.

    Check every device and execute the actual kernel families, rather than merely
    trusting nvidia-smi or wheel installation. Real DDP memory/scaling still needs
    a bounded training qualification on the rental.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise RuntimeError("Requested CUDA GPUs unavailable")
    profile = profiles()[profile_key] if profile_key else None
    devices = []
    for index in range(gpus):
        with torch.cuda.device(index):
            device = torch.cuda.get_device_properties(index)
            capability = list(torch.cuda.get_device_capability(index))
            memory = device.total_memory / 2**30
            if profile and (profile["gpu_name"] not in device.name or
                            capability != profile["compute_capability"] or
                            memory < profile["minimum_vram_gib"]):
                raise RuntimeError(f"GPU {index} does not match {profile_key}: {device.name}, {memory:.1f} GiB, {capability}")
            if capability[0] >= 12 and (not torch.version.cuda or
                    tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8) or
                    "sm_120" not in torch.cuda.get_arch_list()):
                raise RuntimeError("Blackwell requires CUDA >=12.8 and an sm_120 PyTorch wheel; use --hardware-profile 5090-32 --setup")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"GPU {index} does not support BF16")
            parameter = torch.nn.Parameter(torch.randn(128, 128, device=f"cuda:{index}"))
            optimizer = torch.optim.AdamW([parameter], fused=True)
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                projected = parameter @ parameter.T
                q = projected.reshape(1, 2, 128, 64)
                loss = scaled_dot_product_attention(q, q, q, is_causal=True).float().square().mean()
            loss.backward()
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"GPU {index} kernel check produced nonfinite gradients")
            optimizer.step()
            torch.cuda.synchronize(index)
            devices.append({"index": index, "name": device.name, "vram_gib": memory,
                            "compute_capability": capability})
            del loss, q, projected, optimizer, parameter
            torch.cuda.empty_cache()
    print(json.dumps({"torch": str(torch.__version__), "cuda": torch.version.cuda,
                      "devices": devices, "kernel_check": "BF16 matmul, fused SDPA backward, fused AdamW"}), flush=True)
