"""Bounded, synthetic full-model benchmark; never a pretraining checkpoint."""
import argparse
import json
from pathlib import Path
import time

from microllama.common.paths import configure_storage
from microllama.training.cuda import build_model
from microllama.training.objective import chunked_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-length", type=int, default=8192)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--accumulation-steps", type=int, default=1)
    args = p.parse_args()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    root = configure_storage()
    torch.manual_seed(42)
    model = build_model(json.loads(Path("configs/model-338m.json").read_text()), not args.no_checkpointing).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)
    batch = torch.randint(8, 64402, (args.micro_batch_size, args.sequence_length + 1), device="cuda")
    results = []
    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accumulation_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                loss = chunked_loss(model, batch[:, :-1], batch[:, 1:], 256)
            (loss / args.accumulation_steps).backward()
            loss_sum += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        record = {"step": step + 1, "loss": loss_sum / args.accumulation_steps, "grad_norm": float(norm),
                  "tokens_per_second": args.accumulation_steps * args.micro_batch_size * args.sequence_length / (time.perf_counter() - start),
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        results.append(record)
        print(json.dumps(record), flush=True)
    name = f"gpu_benchmark-b{args.micro_batch_size}-a{args.accumulation_steps}-checkpoint{not args.no_checkpointing}.json"
    (root / "logs" / name).write_text(json.dumps({"synthetic": True,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "sequence_length": args.sequence_length, "micro_batch_size": args.micro_batch_size,
        "gradient_checkpointing": not args.no_checkpointing, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
