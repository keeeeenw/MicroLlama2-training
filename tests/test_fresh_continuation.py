"""Exercise the actual CLI handoff in distinct Python processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_data import write_shard
from test_training import TINY


def test_same_context_cli_handoff_preserves_adam_cursor_and_rng(tmp_path):
    root = tmp_path / "data"
    tokens = root / "tokens"
    tokens.mkdir(parents=True)
    write_shard(tokens / "a", [2, 3, 7, 4, 5, 7, 8, 9] * 8)
    meta = json.loads((tokens / "a/manifest.json").read_text())
    (tokens / "manifest.json").write_text(json.dumps({"source_revision": meta["source_revision"],
        "tokenizer_sha256": meta["tokenizer_sha256"], "complete": True, "status": "complete"}))
    (root / "tokenizer").mkdir()
    (root / "tokenizer/source_manifest.json").write_text(json.dumps({
        "tokenizer_sha256": meta["tokenizer_sha256"], "max_token_id": 63,
        "bos_token_id": 1, "eos_token_id": 7, "pad_token_id": 0}))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({**TINY, "bos_token_id": 1, "eos_token_id": 7}))
    old = json.loads(Path("configs/train-8k.json").read_text())
    old.update(model_config=str(model_path), run_name="old", sequence_length=8,
               gradient_accumulation_steps=2, target_tokens=1000, warmup_tokens=16,
               precision="fp32", eval_every=100, save_every=100, loss_chunk_size=4,
               gradient_checkpointing=False)
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old))
    checkpoints = tmp_path / "checkpoints"
    env = {**os.environ, "MICROLLAMA_DATA_ROOT": str(root), "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}

    def run(config, extra):
        result = subprocess.run([sys.executable, "-m", "microllama.training.cuda", "--device", "cpu",
            "--config", str(config), "--streaming-data", "--token-dir", str(tokens),
            "--checkpoint-dir", str(checkpoints), *extra], env=env,
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr

    run(old_path, ["--max-steps", "1"])
    parent = checkpoints / "old/step-00000001"
    state = torch.load(parent / "training_state.pt", weights_only=True)
    # Zero-update handoff proves state preservation before the new objective.
    new = {**old, "run_name": "new", "loss_policy": "document-boundaries-v2",
           "schedule": "constant-after-warmup-v1"}
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(new))
    run(new_path, ["--continue-from", str(parent), "--max-steps", "1"])
    migrated = checkpoints / "new/step-00000001"
    restored = torch.load(migrated / "training_state.pt", weights_only=True)
    assert restored["data"] == state["data"]
    assert restored["step"] == state["step"] and restored["tokens"] == state["tokens"]
    assert restored["lineage"]["historical_target_policy"] == "legacy-unmasked-v1"
    assert torch.equal(restored["torch_rng"], state["torch_rng"])
    for key, value in state["optimizer"]["state"].items():
        for field, original in value.items():
            actual = restored["optimizer"]["state"][key][field]
            if isinstance(original, torch.Tensor):
                assert torch.equal(original, actual)
            else:
                assert original == actual
    run(new_path, ["--resume", str(migrated), "--max-steps", "2"])
    advanced = torch.load(checkpoints / "new/step-00000002/training_state.pt", weights_only=True)
    assert advanced["data"]["cursor"] == state["data"]["cursor"] + 2
    assert 0 < advanced["tokens"] - state["tokens"] < 16
    assert advanced["input_positions"] - restored["input_positions"] == 16
    assert (checkpoints / "new/step-00000002/checkpoint_manifest.json").is_file()
