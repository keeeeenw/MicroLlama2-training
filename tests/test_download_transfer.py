import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from microllama.data_io.download import main as download
from microllama.data_io.transfer import copy_item, verified_destination


def fixture_hub(tmp_path, monkeypatch):
    import huggingface_hub
    snapshot = tmp_path / "hf-cache/snapshots/fixture"
    files = []
    for index in range(2):
        blob = tmp_path / f"blob-{index}"
        blob.write_bytes(bytes([index + 1]) * 9000)
        path = snapshot / f"sample/10BT/{index}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(blob)
        files.append(SimpleNamespace(rfilename=str(path.relative_to(snapshot)), size=9000,
                    lfs=SimpleNamespace(sha256=hashlib.sha256(blob.read_bytes()).hexdigest())))

    class Api:
        def dataset_info(self, repo_id, revision, files_metadata):
            assert files_metadata and len(revision) == 40
            return SimpleNamespace(sha="fixture", siblings=files)

    calls = []

    def get_snapshot(**kwargs):
        # Preparation can see the official snapshot location before the blocking download.
        manifest = json.loads((root / "raw/fineweb-edu-10bt/manifest.json").read_text())
        assert manifest["data_dir"].endswith("datasets--HuggingFaceFW--fineweb-edu/snapshots/fixture")
        assert not manifest["download_complete"] and not manifest["verified"]
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", get_snapshot)
    root = tmp_path / "project-data"
    monkeypatch.setenv("MICROLLAMA_DATA_ROOT", str(root))
    return root, snapshot, calls


def test_default_uses_parallel_hf_cache_without_transfer(tmp_path, monkeypatch):
    root, snapshot, calls = fixture_hub(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HOME", "/user-selected/hf-cache")
    monkeypatch.setattr(sys, "argv", ["download", "--workers", "3"])
    download()
    assert calls[0]["max_workers"] == 3 and calls[0]["cache_dir"] is None
    assert "local_dir" not in calls[0]
    assert calls[0]["allow_patterns"] == ["sample/10BT/0.parquet", "sample/10BT/1.parquet"]
    assert not (root / "raw/fineweb-edu-10bt/sample").exists()
    manifest = json.loads((root / "raw/fineweb-edu-10bt/manifest.json").read_text())
    paths = [Path(manifest["data_dir"]) / item["path"] for item in manifest["files"]]
    assert paths[0].is_symlink() and manifest["data_dir"] == str(snapshot)
    assert manifest["verified"] and manifest["transfer_complete"]
    from microllama.common.paths import configure_storage
    configure_storage()
    import os
    assert os.environ["HF_HOME"] == "/user-selected/hf-cache"


def test_external_transfer_materializes_links_and_reuses_receipts(tmp_path, monkeypatch):
    root, snapshot, calls = fixture_hub(tmp_path, monkeypatch)
    output = tmp_path / "external"
    monkeypatch.setattr(sys, "argv", ["download", "--output", str(output)])
    download()
    manifest = json.loads((root / "raw/fineweb-edu-10bt/manifest.json").read_text())
    paths = [Path(manifest["data_dir"]) / item["path"] for item in manifest["files"]]
    assert paths[0].parent.parent.parent == output
    assert not paths[0].is_symlink() and (snapshot / "sample/10BT/0.parquet").is_symlink()
    assert manifest["transfer_complete"]
    from microllama.data_io import transfer as module
    monkeypatch.setattr(module, "checksum", lambda *args: pytest.fail("Unchanged verified copy was rehashed"))
    assert verified_destination(output, manifest["files"][0])


def test_native_partial_resume_and_corrupted_receipt_detection(tmp_path):
    data = bytes(range(256)) * 100
    source = tmp_path / "source"
    source.write_bytes(data)
    item = {"path": "sample/10BT/a.parquet", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    raw = tmp_path / "external"
    target = raw / item["path"]
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(".parquet.copy-partial")
    partial.write_bytes(b"damaged prefix")
    copy_item(item, source, raw)
    assert target.read_bytes() == data and source.exists() and not partial.exists()
    target.write_bytes(b"x" * len(data))
    assert not verified_destination(raw, item)
    copy_item(item, source, raw)
    assert target.read_bytes() == data
