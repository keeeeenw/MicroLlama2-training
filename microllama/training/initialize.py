"""Save the full, randomly initialized model under the checkpoint data root."""
import argparse
import json
from pathlib import Path

from ..common.paths import configure_storage
from .cuda import build_model, copy_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model-338m.json")
    args = p.parse_args()
    root = configure_storage()
    import torch
    mc = json.loads(Path(args.config).read_text())
    tokenizer = json.loads((root / "tokenizer" / "source_manifest.json").read_text())
    if mc["vocab_size"] <= tokenizer["max_token_id"]:
        raise RuntimeError("Tokenizer vocabulary exceeds embedding size")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if mc[key] != tokenizer[key]:
            raise RuntimeError(f"Special token mismatch: {key}")
    target = root / "checkpoints" / "initial-338m"
    if target.exists():
        raise RuntimeError(f"Already initialized: {target}")
    torch.manual_seed(42)
    model = build_model(mc)
    temporary = target.with_name(target.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temporary, safe_serialization=True)
    copy_tokenizer(root, temporary)
    (temporary / "initialization.json").write_text(json.dumps({"seed": 42, "trained": False,
        "training_tokens": 0, "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_sha256": tokenizer["tokenizer_sha256"]}, indent=2) + "\n")
    temporary.rename(target)
    print(f"Saved untrained initial checkpoint: {target}", flush=True)


if __name__ == "__main__":
    main()
