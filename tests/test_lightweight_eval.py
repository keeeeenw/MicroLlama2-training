import json
from pathlib import Path

import pytest

from microllama.data_io.corpus import content_hash
from microllama.evaluation import lightweight, fixtures
from scripts.evaluation.watch_checkpoints import evaluation_complete


def test_fixed_fixture_uses_full_document_split_and_content_hash(tmp_path, monkeypatch):
    root = tmp_path / "data"
    raw = root / "raw/fineweb-edu-10bt"
    tokenizer_dir = root / "tokenizer"
    raw.mkdir(parents=True)
    tokenizer_dir.mkdir()
    shard = raw / "small.parquet"
    shard.write_bytes(b"fixture")
    source_manifest = {"revision": "rev1", "files": [
        {"path": "small.parquet", "bytes": shard.stat().st_size, "sha256": "source-hash"},
    ]}
    (raw / "manifest.json").write_text(json.dumps(source_manifest))
    (tokenizer_dir / "source_manifest.json").write_text(json.dumps({"tokenizer_sha256": "tokenizer-hash"}))
    original = "held-out full text" * 100
    seen = []
    monkeypatch.setattr(fixtures, "sha256", lambda path: "source-hash")
    monkeypatch.setattr(fixtures, "split_for", lambda text: seen.append(text) or "validation")
    monkeypatch.setattr(fixtures, "documents", lambda path: iter([{"id": "doc-1", "text": original}]))
    settings = {"documents": 1, "max_characters_per_document": 12}
    fixture, path = fixtures.fixed_fixture(root, settings, tmp_path / "fixtures")

    assert seen == [original]
    assert fixture["samples"][0]["text"] == original[:12]
    assert fixture["samples"][0]["content_hash"] == content_hash(original)
    assert fixture["samples"][0]["truncated"] is True
    assert fixture["source_sha256"] == "source-hash"
    assert fixture["identity"]["source_revision"] == "rev1"
    assert fixture["identity_sha256"].startswith(path.stem.removeprefix("fixture-"))

    # The persisted fixture is reused, while a changed official source hash creates a new identity.
    seen.clear()
    cached, cached_path = fixtures.fixed_fixture(root, settings, tmp_path / "fixtures")
    assert cached == fixture
    assert cached_path == path
    assert seen == []
    source_manifest["files"][0]["sha256"] = "new-source-hash"
    (raw / "manifest.json").write_text(json.dumps(source_manifest))
    monkeypatch.setattr(fixtures, "sha256", lambda path: "new-source-hash")
    _, changed_path = fixtures.fixed_fixture(root, settings, tmp_path / "fixtures")
    assert changed_path != path


def test_repetition_rate_and_cpu_config_are_fixed():
    settings = json.loads(Path("configs/evaluation/lightweight.json").read_text())
    assert settings["cpu_threads"] == 2
    assert lightweight.repetition_rate([1, 2, 3, 4, 1, 2, 3, 4]) == pytest.approx(0.2)
    assert lightweight.repetition_rate([1, 2, 3]) == 0


@pytest.mark.parametrize("payload,expected", [
    ({"complete": True, "wandb_status": "uploaded"}, True),
    ({"complete": False, "wandb_status": "failed"}, False),
    ({"wandb_status": "pending"}, False),
])
def test_watcher_only_skips_fully_completed_reports(tmp_path, payload, expected):
    output = tmp_path / "result.json"
    output.write_text(json.dumps(payload))
    assert evaluation_complete(output) is expected


def test_watcher_treats_invalid_report_as_incomplete(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("{")
    assert evaluation_complete(output) is False
