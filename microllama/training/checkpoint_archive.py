"""Archive completed training checkpoints without touching the active run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time


_CHECKPOINT_NAME = re.compile(r"^step-(\d+)$")
_RECEIPT = ".archive-receipt.json"
_CACHE = ".archive-cache.json"


def atomic_json(path, value):
    """Write JSON durably, then publish it with an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_inventory(directory):
    """Return SHA-256 and byte count for every regular file below a checkpoint."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Checkpoint contains a symlink: {path}")
        if not path.is_file() or path.relative_to(directory).as_posix() == _RECEIPT:
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        records.append({"path": path.relative_to(directory).as_posix(),
                        "size": size, "sha256": digest.hexdigest()})
    return records


def file_identities(directory):
    """Return stable stat identities for regular files without reading contents."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Checkpoint contains a symlink: {path}")
        if path.relative_to(directory).as_posix() == _RECEIPT:
            continue
        try:
            info = path.stat(follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError(f"Checkpoint changed while scanning: {path}") from error
        if not stat.S_ISREG(info.st_mode):
            continue
        records.append({"path": path.relative_to(directory).as_posix(), "device": info.st_dev,
                        "inode": info.st_ino, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                        "ctime_ns": info.st_ctime_ns})
    return records


def inventory_fingerprint(inventory):
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_alive(pid):
    """Check Linux process state, treating zombies as stopped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().split()[2] != "Z"
        except (IndexError, OSError):
            return False
    return True


def published_checkpoints(run_directory):
    run_directory = Path(run_directory)
    checkpoints = []
    for path in run_directory.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def _latest_checkpoint(run_directory):
    latest_path = Path(run_directory) / "latest.json"
    latest = json.loads(latest_path.read_text())
    pointed = Path(latest["path"]).resolve(strict=True)
    if pointed.parent != Path(run_directory).resolve() or not _CHECKPOINT_NAME.fullmatch(pointed.name):
        raise RuntimeError(f"Latest checkpoint escapes its run directory: {latest_path}")
    if not pointed.is_dir() or pointed.is_symlink():
        raise RuntimeError(f"Latest checkpoint is not a published directory: {pointed}")
    return pointed, latest


def _load_cache(source):
    path = Path(source) / _CACHE
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        cache = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # The cache is only an optimization; discard it and perform full hashes.
        return {"version": 1, "entries": {}}
    if (not isinstance(cache, dict) or cache.get("version") != 1
            or not isinstance(cache.get("entries"), dict)):
        return {"version": 1, "entries": {}}
    return cache


def _write_cache(source, cache):
    atomic_json(Path(source) / _CACHE, cache)


def _cache_key(source, checkpoint, destination):
    return f"{Path(checkpoint).parent.name}/{Path(checkpoint).name}@{Path(destination).resolve()}"


def _cached_match(cache, key, receipt, source_stats, destination_stats, source_checkpoint, run_name, step):
    entry = cache["entries"].get(key)
    return bool(isinstance(entry, dict) and isinstance(receipt, dict)
                and receipt.get("source") == str(source_checkpoint)
                and receipt.get("run") == run_name and receipt.get("step") == step
                and entry.get("source_fingerprint") == receipt.get("source_fingerprint")
                and receipt.get("source_fingerprint") == inventory_fingerprint(receipt.get("files", []))
                and entry.get("source_stats") == source_stats == receipt.get("source_stats")
                and entry.get("destination_stats") == destination_stats == receipt.get("destination_stats")
                and entry.get("files") == receipt.get("files")
                and [f["path"] for f in receipt.get("files", [])] == [f["path"] for f in source_stats]
                and all(file["size"] == identity["size"]
                        for file, identity in zip(receipt.get("files", []), source_stats))
                and all(file["size"] == identity["size"]
                        for file, identity in zip(receipt.get("files", []), destination_stats)))


def _write_receipt(final, source_checkpoint, run_name, step, inventory, source_stats, destination_stats):
    fingerprint = inventory_fingerprint(inventory)
    receipt = {"source": str(source_checkpoint), "run": run_name, "step": step,
               "source_fingerprint": fingerprint, "files": inventory,
               "source_stats": source_stats, "destination_stats": destination_stats,
               "file_count": len(inventory), "byte_count": sum(f["size"] for f in inventory)}
    atomic_json(Path(final) / _RECEIPT, receipt)
    _fsync_tree(final)
    return receipt


def _store_cache(cache, key, receipt):
    cache["entries"][key] = {"source_fingerprint": receipt["source_fingerprint"],
                              "files": receipt["files"], "source_stats": receipt["source_stats"],
                              "destination_stats": receipt["destination_stats"]}


def _fsync_tree(directory):
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Archive contains a symlink: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = (p for p in directory.rglob("*") if p.is_dir())
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fsync_directory(path)
    fsync_directory(directory)


def _update_latest(destination_run, final_directory, step, tokens=None):
    latest_path = Path(destination_run) / "latest.json"
    if latest_path.exists():
        current = json.loads(latest_path.read_text())
        if int(current.get("step", -1)) > step:
            return
    value = {"path": str(Path(final_directory).resolve()), "step": step}
    if tokens is not None:
        value["tokens"] = tokens
    if latest_path.exists() and json.loads(latest_path.read_text()) == value:
        return
    atomic_json(latest_path, value)


def archive_checkpoint(source_checkpoint, destination_run, *, run_name=None, tokens=None,
                       progress=None, _cache=None, _cache_root=None):
    """Copy, hash, fsync, and atomically publish one completed checkpoint.

    An interrupted rsync leaves its ``.partial`` directory for the next call to
    resume. Existing published archives are accepted only after receipt and hash
    validation against the immutable source.
    """
    source_checkpoint = Path(source_checkpoint)
    if source_checkpoint.is_symlink():
        raise RuntimeError(f"Checkpoint path is a symlink: {source_checkpoint}")
    source_checkpoint = source_checkpoint.resolve(strict=True)
    match = _CHECKPOINT_NAME.fullmatch(source_checkpoint.name)
    if not match or not source_checkpoint.is_dir():
        raise ValueError(f"Not a published checkpoint directory: {source_checkpoint}")
    step = int(match.group(1))
    run_name = run_name or source_checkpoint.parent.name
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        raise ValueError(f"Invalid run name: {run_name!r}")
    destination_run = Path(destination_run)
    if destination_run.is_symlink():
        raise RuntimeError(f"Destination run directory is a symlink: {destination_run}")
    destination_run.mkdir(parents=True, exist_ok=True)
    destination_run = destination_run.resolve()
    final = destination_run / source_checkpoint.name
    partial = destination_run / (source_checkpoint.name + ".partial")
    cache_root = Path(_cache_root) if _cache_root is not None else source_checkpoint.parent.parent
    cache = _cache if _cache is not None else _load_cache(cache_root)
    cache_key = _cache_key(cache_root, source_checkpoint, final)
    if final.exists():
        if final.is_symlink() or not final.is_dir():
            raise RuntimeError(f"Archive target is not a directory: {final}")
        receipt_path = final / _RECEIPT
        if receipt_path.is_symlink():
            raise RuntimeError(f"Archive receipt is a symlink: {receipt_path}")
        receipt = None
        if receipt_path.is_file():
            try:
                candidate = json.loads(receipt_path.read_text())
                receipt = candidate if isinstance(candidate, dict) else None
            except json.JSONDecodeError:
                # A damaged receipt is replaceable only after full payload hashes match.
                receipt = None
        source_stats = file_identities(source_checkpoint)
        destination_stats_before = file_identities(final)
        if _cached_match(cache, cache_key, receipt, source_stats, destination_stats_before,
                         source_checkpoint, run_name, step):
            file_paths = [entry["path"] for entry in receipt["files"]]
            if (file_paths != [entry["path"] for entry in source_stats]
                    or file_paths != [entry["path"] for entry in destination_stats_before]):
                raise RuntimeError(f"Cached archive file list mismatch: {final}")
            _update_latest(destination_run, final, step, tokens)
            return {"path": str(final), "files": len(receipt["files"]),
                    "bytes": sum(f["size"] for f in receipt["files"]),
                    "source_fingerprint": receipt["source_fingerprint"],
                    "inventory": receipt["files"],
                    "source_stats": source_stats, "destination_stats": destination_stats_before,
                    "already_archived": True}

        source_inventory = file_inventory(source_checkpoint)
        if file_identities(source_checkpoint) != source_stats:
            raise RuntimeError(f"Source checkpoint changed while verifying archive: {source_checkpoint}")
        destination_inventory = file_inventory(final)
        if source_inventory != destination_inventory:
            raise RuntimeError(f"Existing archive differs from source; refusing to overwrite: {final}")
        destination_stats = file_identities(final)
        if destination_stats != destination_stats_before:
            raise RuntimeError(f"Archive changed while verifying its contents: {final}")
        receipt = _write_receipt(final, source_checkpoint, run_name, step, source_inventory,
                                 source_stats, destination_stats)
        _store_cache(cache, cache_key, receipt)
        _write_cache(cache_root, cache)
        _update_latest(destination_run, final, step, tokens)
        return {"path": str(final), "files": len(source_inventory),
                "bytes": sum(f["size"] for f in source_inventory),
                "source_fingerprint": receipt["source_fingerprint"],
                "inventory": source_inventory,
                "source_stats": receipt["source_stats"], "destination_stats": receipt["destination_stats"],
                "already_archived": True}
    if final.is_symlink():
        raise RuntimeError(f"Archive target is a symlink: {final}")
    if partial.is_symlink():
        raise RuntimeError(f"Partial archive target is a symlink: {partial}")
    partial.mkdir(exist_ok=True)
    source_stats = file_identities(source_checkpoint)
    source_inventory = file_inventory(source_checkpoint)
    if file_identities(source_checkpoint) != source_stats:
        raise RuntimeError(f"Source checkpoint changed while hashing: {source_checkpoint}")
    source_fingerprint = inventory_fingerprint(source_inventory)
    if progress:
        progress(current_checkpoint=source_checkpoint.name, phase="copying",
                 file_count=len(source_inventory), byte_count=sum(f["size"] for f in source_inventory))

    command = ["rsync", "--archive", "--partial", "--checksum", "--delete", "--",
               str(source_checkpoint) + "/", str(partial) + "/"]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"rsync exited {result.returncode}"
        raise RuntimeError(f"rsync failed for {source_checkpoint}: {detail}")

    after_copy = file_inventory(source_checkpoint)
    source_stats_after = file_identities(source_checkpoint)
    if inventory_fingerprint(after_copy) != source_fingerprint or source_stats_after != source_stats:
        raise RuntimeError(f"Source checkpoint changed during archive: {source_checkpoint}")
    destination_stats_before = file_identities(partial)
    target_inventory = file_inventory(partial)
    if target_inventory != source_inventory:
        raise RuntimeError(f"Archive contents do not match source checkpoint: {partial}")
    destination_stats = file_identities(partial)
    if destination_stats != destination_stats_before:
        raise RuntimeError(f"Archive changed while verifying its contents: {partial}")
    if [entry["path"] for entry in source_inventory] != [entry["path"] for entry in destination_stats]:
        raise RuntimeError(f"Archive stat inventory does not match source: {partial}")
    receipt = _write_receipt(partial, source_checkpoint, run_name, step, source_inventory,
                             source_stats, destination_stats)
    partial.rename(final)
    fsync_directory(destination_run)
    _store_cache(cache, cache_key, receipt)
    _write_cache(cache_root, cache)
    _update_latest(destination_run, final, step, tokens)
    if progress:
        progress(current_checkpoint=None, phase="checkpoint_archived", archived_checkpoint=str(final))
    return {"path": str(final), "files": len(source_inventory), "bytes": sum(f["size"] for f in source_inventory),
            "source_fingerprint": source_fingerprint, "inventory": source_inventory, "source_stats": source_stats,
            "destination_stats": destination_stats, "already_archived": False}


def _safe_prune(source_checkpoint, source_run, destination_run, verified):
    source_checkpoint = Path(source_checkpoint)
    source_run = Path(source_run).resolve()
    if source_checkpoint.parent.resolve() != source_run or source_checkpoint.is_symlink():
        raise RuntimeError(f"Refusing to prune a path outside the source run: {source_checkpoint}")
    final = Path(destination_run) / source_checkpoint.name
    receipt_path = final / _RECEIPT
    if final.is_symlink() or not receipt_path.is_file():
        raise RuntimeError(f"Refusing to prune an unverified archive: {final}")
    receipt = json.loads(receipt_path.read_text())
    if (file_identities(source_checkpoint) != verified["source_stats"]
            or file_identities(final) != verified["destination_stats"]
            or receipt.get("source_fingerprint") != verified["source_fingerprint"]
            or receipt.get("files") != verified["inventory"]
            or receipt.get("source_stats") != verified["source_stats"]
            or receipt.get("destination_stats") != verified["destination_stats"]):
        raise RuntimeError(f"Refusing to prune a changed or unverified checkpoint: {source_checkpoint}")
    shutil.rmtree(source_checkpoint)


def archive_available(source, destination, *, progress=None, prune=True):
    """Archive every published checkpoint in runs with a valid latest pointer."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and archive destination must be separate directory trees")
    summary = {"runs": 0, "published": 0, "deferred_unpublished": 0,
               "archived": 0, "pruned": 0, "files": 0, "bytes": 0}
    cache = _load_cache(source)
    for source_run in sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink() and (p / "latest.json").is_file()):
        latest_checkpoint, latest_info = _latest_checkpoint(source_run)
        all_checkpoints = published_checkpoints(source_run)
        latest_step = int(_CHECKPOINT_NAME.fullmatch(latest_checkpoint.name).group(1))
        summary["deferred_unpublished"] += sum(1 for step, _ in all_checkpoints if step > latest_step)
        checkpoints = [(step, path) for step, path in all_checkpoints if step <= latest_step]
        if not checkpoints:
            continue
        summary["runs"] += 1
        summary["published"] += len(checkpoints)
        destination_run = destination / source_run.name
        verified = {}
        for step, checkpoint in checkpoints:
            result = archive_checkpoint(checkpoint, destination_run, run_name=source_run.name,
                                        tokens=latest_info.get("tokens") if checkpoint == latest_checkpoint else None,
                                        progress=progress, _cache=cache, _cache_root=source)
            verified[checkpoint] = result
            summary["archived"] += 1
            summary["files"] += result["files"]
            summary["bytes"] += result["bytes"]

        if prune:
            keep = {path for _, path in checkpoints[-2:]}
            keep.add(latest_checkpoint)
            for _, checkpoint in checkpoints:
                if checkpoint in keep:
                    continue
                _safe_prune(checkpoint, source_run, destination_run, verified[checkpoint])
                summary["pruned"] += 1
        if latest_checkpoint not in verified:
            raise RuntimeError(f"Latest checkpoint was not archived: {latest_checkpoint}")
    return summary


def _atomic_status(path, status):
    status["updated_unix"] = time.time()
    atomic_json(path, status)


def run_worker(source, destination, *, train_pid=None, poll_seconds=30):
    """Run the single archive worker until stopped, or until a trainer exits."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    status_path = source / ".archive-status.json"
    lock_path = source / ".archive.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"Another checkpoint archive worker holds {lock_path}") from error

    status = {"pid": os.getpid(), "source": str(source), "destination": str(destination),
              "train_pid": train_pid, "phase": "starting", "error": None}
    retry_delay = poll_seconds
    try:
        while True:
            try:
                status.update(phase="scanning", error=None)
                _atomic_status(status_path, status)
                summary = archive_available(source, destination,
                                            progress=lambda **updates: (_atomic_status(status_path, status.update(updates) or status)))
                status.update(summary)
                trainer_running = train_pid is None or is_alive(train_pid)
                if train_pid is not None and not trainer_running:
                    # Catch a checkpoint whose directory/latest rename completed just as the
                    # first scan finished and the trainer exited.
                    status.update(phase="final_scan")
                    _atomic_status(status_path, status)
                    summary = archive_available(
                        source, destination,
                        progress=lambda **updates: (_atomic_status(status_path, status.update(updates) or status)))
                    status.update(summary)
                    status.update(phase="complete", error=None)
                    _atomic_status(status_path, status)
                    return summary
                status.update(phase="watching", error=None)
                _atomic_status(status_path, status)
                retry_delay = poll_seconds
                time.sleep(poll_seconds)
            except KeyboardInterrupt:
                status.update(phase="interrupted", error=None)
                _atomic_status(status_path, status)
                raise
            except Exception as error:
                status.update(phase="retrying", error=str(error), retry_seconds=retry_delay)
                _atomic_status(status_path, status)
                time.sleep(retry_delay)
                retry_delay = min(max(poll_seconds, retry_delay * 2), 300)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
