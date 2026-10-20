"""Archive immutable local token chunks to TrainData after raw copies finish."""
import argparse
import json
from pathlib import Path
import os
import shutil
import time

from .download import sha256
from ..common.paths import configure_storage
from .prepare import update_summary
from .transfer import atomic_json


def archive_chunk(source, destination):
    meta = json.loads((source / "manifest.json").read_text())
    if destination.exists():
        if json.loads((destination / "manifest.json").read_text()) != meta:
            raise RuntimeError(f"Archive chunk identity mismatch: {destination}")
        for split in ("train", "validation"):
            if sha256(destination / f"{split}.bin") != meta["splits"][split]["sha256"]:
                raise RuntimeError(f"Archive checksum mismatch: {destination}")
        return
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for file in source.iterdir():
        if file.is_file():
            with file.open("rb") as src, (temporary / file.name).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 2**20)
                dst.flush()
                os.fsync(dst.fileno())
    for split in ("train", "validation"):
        if sha256(temporary / f"{split}.bin") != meta["splits"][split]["sha256"]:
            raise RuntimeError(f"Archive copy checksum mismatch: {source}")
    temporary.rename(destination)
    print(f"Archived and verified {destination.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = configure_storage()
    output = root / "tokens"
    output.mkdir(exist_ok=True)
    raw_manifest = root / "raw/fineweb-edu-10bt/manifest.json"
    while True:
        raw = json.loads(raw_manifest.read_text())
        if raw.get("verified") and raw.get("transfer_complete", True):
            break
        time.sleep(10)
    while True:
        summary = json.loads((args.source / "manifest.json").read_text())
        if summary.get("status") == "failed":
            raise RuntimeError("Token producer failed; archive remains incomplete")
        for path in sorted(args.source.glob("*/manifest.json")):
            if not path.parent.name.endswith(".partial") and not (output / path.parent.name).exists():
                archive_chunk(path.parent, output / path.parent.name)
                update_summary(output, {"revision": summary["source_revision"]}, summary["tokenizer_sha256"], False)
        if summary["complete"]:
            for path in sorted(args.source.glob("*/manifest.json")):
                if not path.parent.name.endswith(".partial"):
                    archive_chunk(path.parent, output / path.parent.name)
            if (args.source / ".sources").exists():
                shutil.copytree(args.source / ".sources", output / ".sources", dirs_exist_ok=True)
            atomic_json(output / "manifest.json", summary)
            print("Complete token corpus archived to TrainData; local training cache retained.", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
