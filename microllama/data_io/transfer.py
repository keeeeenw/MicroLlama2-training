"""Optionally copy a verified Hugging Face dataset snapshot to another drive."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ..common.paths import data_root


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def checksum(path, report, phase):
    digest = hashlib.sha256()
    total = 0
    last = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 2**20), b""):
            digest.update(block)
            total += len(block)
            if time.monotonic() - last > 2:
                report(phase=phase, checked_bytes=total)
                last = time.monotonic()
    return digest.hexdigest()


def receipt_path(raw, item):
    return raw / ".verified" / (Path(item["path"]).name + ".json")


def verified_destination(raw, item, report=lambda **kw: None):
    target = raw / item["path"]
    receipt = receipt_path(raw, item)
    if not target.exists() or target.stat().st_size != item["bytes"]:
        return False
    identity = file_identity(target)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["sha256"] == item["sha256"] and saved["file_identity"] == identity:
            return True
    if checksum(target, report, "verifying_existing") != item["sha256"]:
        return False
    with target.open("rb") as handle:
        report(phase="flushing_existing")
        os.fsync(handle.fileno())
    atomic_json(receipt, {"sha256": item["sha256"], "file_identity": file_identity(target)})
    return True


def copy_item(item, staged, raw, report=lambda **kw: None):
    target = raw / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = receipt_path(raw, item)
    receipt.parent.mkdir(exist_ok=True)
    with receipt.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if verified_destination(raw, item, report):
            return
        if not staged.exists() or staged.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Source shard missing or incomplete: {staged}")
        partial = target.with_suffix(".parquet.copy-partial")
        full_partial = partial.exists() and partial.stat().st_size == item["bytes"]
        reusable = full_partial and checksum(partial, report, "verifying_partial") == item["sha256"]
        if full_partial and not reusable:
            partial.unlink()
        if not reusable:
            if shutil.which("rsync") is None:
                raise RuntimeError("Native copying requires rsync; install it or copy manually and use --verify-only")
            # Native delta copying reuses and repairs partials, including corrupt prefixes.
            command = ["rsync", "--copy-links", "--inplace", "--partial", "--no-whole-file",
                       "--progress", "--", str(staged), str(partial)]
            print(f"Native copy {target.name}", flush=True)
            child = subprocess.Popen(command)
            try:
                while child.poll() is None:
                    buffered = partial.stat().st_size if partial.exists() else 0
                    report(phase="writing_or_flushing" if buffered == item["bytes"] else "copying",
                           native_pid=child.pid, buffered_bytes=buffered)
                    time.sleep(0.25)
            except BaseException:
                child.terminate()
                child.wait()
                raise
            if child.returncode:
                raise RuntimeError(f"rsync exited with {child.returncode}; partial copy retained: {partial}")
        report(phase="flushing_partial", buffered_bytes=item["bytes"])
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        if not reusable and checksum(partial, report, "verifying_copy") != item["sha256"]:
            raise RuntimeError(f"Copied shard checksum mismatch: {partial}")
        partial.replace(target)
        atomic_json(receipt, {"sha256": item["sha256"], "file_identity": file_identity(target)})
        # Retain downloaded sources so ongoing tokenization always has fast SSD input.
        report(phase="copied_verified", buffered_bytes=item["bytes"])
        print(f"Durably copied and SHA-256 verified {target.name}", flush=True)


def transfer(manifest, source, destination, manifest_path, verify_only=False):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".transfer.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another dataset transfer is running")
        for index, item in enumerate(manifest["files"], 1):
            print(f"File {index}/{len(manifest['files'])}: {item['path']}", flush=True)
            if verify_only:
                if not verified_destination(destination, item):
                    raise RuntimeError(f"Manual copy missing or checksum mismatch: {item['path']}")
            else:
                copy_item(item, source / item["path"], destination)
        manifest.update(data_dir=str(destination), verified=True, download_complete=True, transfer_complete=True)
        atomic_json(manifest_path, manifest)
        if destination / "manifest.json" != manifest_path.resolve():
            atomic_json(destination / "manifest.json", manifest)
        print(f"All files safely copied and verified: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Snapshot directory; defaults to the downloaded dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Destination folder on the external drive")
    parser.add_argument("--verify-only", action="store_true", help="Register and verify files already copied manually")
    args = parser.parse_args()
    manifest_path = data_root() / "raw/fineweb-edu-10bt/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = args.source or Path(manifest.get("data_dir", manifest_path.parent))
    transfer(manifest, source, args.output, manifest_path, args.verify_only)


if __name__ == "__main__":
    main()
