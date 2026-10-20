import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from microllama.training import checkpoint_archive as archive


def make_checkpoint(run, step, *, model=b"model-weights", state=b"training-state"):
    checkpoint = run / f"step-{step:08d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(model)
    (checkpoint / "training_state.pt").write_bytes(state)
    (checkpoint / "nested").mkdir()
    (checkpoint / "nested" / "run_config.json").write_text('{"seed":42}\n')
    return checkpoint


def set_latest(run, checkpoint, step, tokens):
    (run / "latest.json").write_text(json.dumps({"path": str(checkpoint), "step": step, "tokens": tokens}))


def test_archive_hashes_publishes_latest_and_prunes_only_verified_old_checkpoint(tmp_path):
    source = tmp_path / "nvme" / "checkpoints"
    source_run = source / "run-a"
    source_run.mkdir(parents=True)
    checkpoints = [make_checkpoint(source_run, step) for step in (1000, 2000, 3000)]
    set_latest(source_run, checkpoints[-1], 3000, 393_216_000)
    destination = tmp_path / "TrainData" / "checkpoints"
    expected_bytes = sum(p.stat().st_size for cp in checkpoints for p in cp.rglob("*") if p.is_file())

    result = archive.archive_available(source, destination)

    assert result == {"runs": 1, "published": 3, "deferred_unpublished": 0,
                      "archived": 3, "pruned": 1,
                      "files": 9, "bytes": expected_bytes}
    assert not checkpoints[0].exists()
    assert checkpoints[1].is_dir() and checkpoints[2].is_dir()
    final = destination / "run-a" / checkpoints[2].name
    assert archive.file_inventory(final) == archive.file_inventory(checkpoints[2])
    receipt = json.loads((final / archive._RECEIPT).read_text())
    assert receipt["source_fingerprint"] == archive.inventory_fingerprint(archive.file_inventory(checkpoints[2]))
    latest = json.loads((destination / "run-a" / "latest.json").read_text())
    assert latest == {"path": str(final.resolve()), "step": 3000, "tokens": 393_216_000}


def test_interrupted_partial_is_resumed_and_repaired(tmp_path):
    source_run = tmp_path / "source" / "run-a"
    source_run.mkdir(parents=True)
    checkpoint = make_checkpoint(source_run, 1000, model=b"complete model weights")
    destination_run = tmp_path / "archive" / "run-a"
    partial = destination_run / (checkpoint.name + ".partial")
    (partial / "nested").mkdir(parents=True)
    source_model = checkpoint / "model.safetensors"
    corrupt = b"X" + source_model.read_bytes()[1:]
    (partial / "model.safetensors").write_bytes(corrupt)
    os.utime(partial / "model.safetensors",
             ns=(source_model.stat().st_atime_ns, source_model.stat().st_mtime_ns))
    assert (partial / "model.safetensors").stat().st_size == source_model.stat().st_size
    assert (partial / "model.safetensors").stat().st_mtime_ns == source_model.stat().st_mtime_ns
    (partial / "nested" / "run_config.json").write_text("stale file")

    result = archive.archive_checkpoint(checkpoint, destination_run)

    final = Path(result["path"])
    assert final.is_dir()
    assert not partial.exists()
    assert archive.file_inventory(final) == archive.file_inventory(checkpoint)
    assert not (final / "nested" / "run_config.json").read_text() == "stale file"


def test_failed_copy_keeps_unarchived_local_checkpoint_and_partial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source_run = source / "run-a"
    source_run.mkdir(parents=True)
    older, latest = [make_checkpoint(source_run, step) for step in (1000, 2000)]
    set_latest(source_run, latest, 2000, 262_144_000)
    destination = tmp_path / "archive"
    real_run = subprocess.run

    def interrupted(command, **kwargs):
        if str(latest) + "/" in command[-2]:
            partial = Path(command[-1].rstrip("/"))
            partial.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(latest / "model.safetensors", partial / "model.safetensors")
            (partial / "model.safetensors").write_bytes(b"truncated")
            return subprocess.CompletedProcess(command, 12, "", "simulated interruption")
        return real_run(command, **kwargs)

    monkeypatch.setattr(archive.subprocess, "run", interrupted)
    with pytest.raises(RuntimeError, match="rsync failed"):
        archive.archive_available(source, destination)

    assert older.is_dir()
    assert latest.is_dir()
    assert (destination / "run-a" / (latest.name + ".partial") / "model.safetensors").read_bytes() == b"truncated"


def test_adopts_existing_complete_archive_and_refuses_corrupt_archive(tmp_path):
    source = tmp_path / "source"
    source_run = source / "run-a"
    source_run.mkdir(parents=True)
    checkpoint = make_checkpoint(source_run, 1000)
    set_latest(source_run, checkpoint, 1000, 131_072_000)
    destination = tmp_path / "archive"
    final = destination / "run-a" / checkpoint.name
    final.parent.mkdir(parents=True)
    shutil.copytree(checkpoint, final)

    result = archive.archive_available(source, destination)

    assert result["archived"] == 1
    assert (final / archive._RECEIPT).is_file()
    receipt = json.loads((final / archive._RECEIPT).read_text())
    assert receipt["source_fingerprint"] == archive.inventory_fingerprint(archive.file_inventory(checkpoint))

    (final / "model.safetensors").write_bytes(b"corrupt existing archive")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        archive.archive_available(source, destination)
    assert (final / "model.safetensors").read_bytes() == b"corrupt existing archive"


def test_unchanged_verified_archive_skips_content_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source_run = source / "run-a"
    source_run.mkdir(parents=True)
    checkpoint = make_checkpoint(source_run, 1000)
    set_latest(source_run, checkpoint, 1000, 131_072_000)
    destination = tmp_path / "archive"
    archive.archive_available(source, destination)

    def no_content_hashes(_):
        raise AssertionError("unchanged archive poll should use stat identities")

    monkeypatch.setattr(archive, "file_inventory", no_content_hashes)
    result = archive.archive_available(source, destination)

    assert result["archived"] == 1
    assert (source / archive._CACHE).is_file()


def test_checkpoint_ahead_of_latest_pointer_is_deferred(tmp_path):
    source = tmp_path / "source"
    source_run = source / "run-a"
    source_run.mkdir(parents=True)
    current = make_checkpoint(source_run, 1000)
    not_yet_published = make_checkpoint(source_run, 2000)
    set_latest(source_run, current, 1000, 131_072_000)
    destination = tmp_path / "archive"

    result = archive.archive_available(source, destination)

    assert result["published"] == 1
    assert result["deferred_unpublished"] == 1
    assert (destination / "run-a" / current.name).is_dir()
    assert not (destination / "run-a" / not_yet_published.name).exists()
    assert not_yet_published.is_dir()


def test_worker_writes_status_and_lock_under_source(tmp_path, monkeypatch):
    source = tmp_path / "nvme" / "checkpoints"
    source.mkdir(parents=True)
    destination = tmp_path / "archive"
    monkeypatch.setattr(archive, "is_alive", lambda _pid: False)

    archive.run_worker(source, destination, train_pid=12345, poll_seconds=0.01)

    assert (source / ".archive-status.json").is_file()
    assert (source / ".archive.lock").is_file()
