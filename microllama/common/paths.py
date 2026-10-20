import os
import json
from pathlib import Path


def data_root():
    if os.environ.get("MICROLLAMA_DATA_ROOT"):
        return Path(os.environ["MICROLLAMA_DATA_ROOT"]).expanduser()
    local = Path(__file__).resolve().parents[2] / ".runtime/storage.json"
    if local.exists():
        return Path(json.loads(local.read_text())["data_root"])
    return Path(__file__).resolve().parents[2] / "TrainData"


def configure_storage():
    root = data_root()
    for name in ("raw", "tokenizer", "tokens", "checkpoints", "logs", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_root(root=None):
    """Optional fast checkpoint staging; the data root remains the archive home."""
    root = root or data_root()
    if os.environ.get("MICROLLAMA_CHECKPOINT_ROOT"):
        return Path(os.environ["MICROLLAMA_CHECKPOINT_ROOT"]).expanduser().resolve()
    local = Path(__file__).resolve().parents[2] / ".runtime/storage.json"
    if local.exists():
        settings = json.loads(local.read_text())
        if Path(settings["data_root"]).resolve() == root.resolve() and settings.get("checkpoint_root"):
            return Path(settings["checkpoint_root"]).expanduser().resolve()
    return root / "checkpoints"


def training_log_root(root=None):
    """Place hot training metrics on the same fast volume as checkpoints."""
    root = root or data_root()
    local = Path(__file__).resolve().parents[2] / ".runtime/storage.json"
    if local.exists():
        settings = json.loads(local.read_text())
        if Path(settings["data_root"]).resolve() == root.resolve() and settings.get("training_log_root"):
            return Path(settings["training_log_root"]).expanduser().resolve()
    return root / "logs"
