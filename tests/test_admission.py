import json
import hashlib
from pathlib import Path

import pytest

from microllama.data_io.admission import (
    AdmissionRegistry,
    build_selection,
    content_split,
    select_files,
    source_identity,
    source_key,
    validate_raw_manifest,
    validate_storage_budget,
    main,
)
from microllama.data_io.corpus import content_hash, split_for


def identity(repo, revision="rev-1", config="sample"):
    return {"repo": repo, "revision": revision, "config": config}


def raw_manifest(repo="owner/data", revision="rev-1", files=None):
    files = files or [
        {"path": "part/a.parquet", "bytes": 6, "sha256": "a" * 64},
        {"path": "part/b.parquet", "bytes": 5, "sha256": "b" * 64},
        {"path": "part/c.parquet", "bytes": 4, "sha256": "c" * 64},
    ]
    return {"repo": repo, "revision": revision, "config": "default",
            "total_bytes": sum(item["bytes"] for item in files), "files": files}


def row(doc_id, text, **extra):
    return {"source_doc_id": doc_id, "text": text,
            "utf8_bytes": len(text.encode("utf-8")), **extra}


def test_source_ids_are_namespaced_but_exact_content_is_global(tmp_path):
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        registry.ingest_documents(identity("owner/one"), [row("shared-id", "content from one")])
        registry.ingest_documents(identity("owner/two"), [row("shared-id", "different content")])
        assert registry.db.execute("SELECT COUNT(*) FROM documents WHERE source_doc_id='shared-id'").fetchone()[0] == 2

        with pytest.raises(RuntimeError, match="exact duplicate"):
            registry.ingest_documents(identity("owner/three"), [row("another-id", "content from one")])
        assert registry.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 2


def test_candidate_duplicates_roll_back_atomically(tmp_path):
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        with pytest.raises(RuntimeError, match="exact duplicate"):
            registry.ingest_documents(identity("owner/data"), [
                row("id-1", "same normalized content"),
                row("id-2", " same   normalized content "),
            ])
        assert registry.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert registry.db.execute("SELECT COUNT(*) FROM content_groups").fetchone()[0] == 0


def test_deterministic_duplicate_group_split_matches_existing_recipe():
    variants = ["duplicate group with unicode café", " duplicate   group with unicode café "]
    hashes = {content_hash(text) for text in variants}
    assert len(hashes) == 1
    assigned = {content_split(value) for value in hashes}
    assert len(assigned) == 1
    assert assigned == {split_for(variants[0])}
    assert content_split(next(iter(hashes))) == content_split(next(iter(hashes)))


def test_existing_train_or_validation_content_cannot_become_sealed_test(tmp_path):
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        train_text = "previously admitted training document"
        validation_text = next(
            text for i in range(10000)
            if (text := f"previously admitted validation {i}")
            and content_split(content_hash(text)) == "validation")
        registry.ingest_documents(identity("owner/train"), [row("train-id", train_text)])
        registry.ingest_documents(identity("owner/validation"), [row("val-id", validation_text)])
        for doc_id, text in (("test-train-copy", train_text), ("test-val-copy", validation_text)):
            with pytest.raises(RuntimeError, match="Held-out overlap"):
                registry.ingest_documents(identity("owner/test"), [row(doc_id, text)], mode="sealed_test",
                    sealed_test_selection_id="would-seal")
        assert registry.db.execute("SELECT value FROM registry_meta WHERE key='sealed_test_selection_id'").fetchone() is None


def test_sealed_test_blocks_future_training_and_cannot_be_replaced(tmp_path):
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        test_text = "a fixed evaluation-only document"
        registry.ingest_documents(identity("owner/test"), [row("test-id", test_text)], mode="sealed_test",
                                  sealed_test_selection_id="selection-one")
        with pytest.raises(RuntimeError, match="Held-out overlap"):
            registry.ingest_documents(identity("owner/train"), [row("train-copy", test_text)])
        with pytest.raises(RuntimeError, match="different sealed test selection"):
            registry.ingest_documents(identity("owner/new-test"), [row("other-id", "another test")],
                                      mode="sealed_test", sealed_test_selection_id="selection-two")
        assert registry.db.execute("SELECT COUNT(*) FROM documents WHERE split='test'").fetchone()[0] == 1


def test_prior_train_validation_and_test_sidecars_import_as_provenance(tmp_path):
    token_root = tmp_path / "tokens"
    chunk = token_root / "source-000000"
    chunk.mkdir(parents=True)
    meta = {"source_revision": "rev-1", "source": "source.parquet", "tokenizer_sha256": "tok",
            "splits": {split: {} for split in ("train", "validation", "test")}}
    (chunk / "manifest.json").write_text(json.dumps(meta))
    texts = {"train": "an existing train row", "validation": "an existing validation row",
             "test": "an existing test row"}
    for split, text in texts.items():
        (chunk / f"{split}.documents.jsonl").write_text(json.dumps({
            "id": f"id-{split}", "content_hash": content_hash(text), "utf8_bytes": len(text),
            "offset": 0, "tokens": 4}) + "\n")
    manifest = {"repo": "owner/prior", "revision": "rev-1", "config": "sample"}
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        counts = registry.import_sidecars(token_root, identity=source_identity(manifest))
        assert counts["sidecars"] == 3
        assert counts["records"] == 3
        assert counts["train_documents"] == 1
        assert counts["validation_documents"] == 1
        assert counts["test_documents"] == 1
        assert registry.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3


def test_raw_byte_budget_and_free_space_reserve_are_enforced():
    manifest = raw_manifest()
    _, files, total = validate_raw_manifest(manifest)
    selected, skipped = select_files(files, byte_budget=10)
    assert [item["path"] for item in selected] == ["part/a.parquet", "part/c.parquet"]
    assert sum(item["bytes"] for item in selected) == 10
    assert [item["path"] for item in skipped] == ["part/b.parquet"]
    assert total == 15

    with pytest.raises(RuntimeError, match="over byte budget"):
        validate_storage_budget(selected_bytes=11, byte_budget=10, free_bytes=100, reserve_bytes=1)
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        validate_storage_budget(selected_bytes=8, byte_budget=10, free_bytes=8, reserve_bytes=1)


def test_selection_is_pinned_deterministic_and_immutable(tmp_path):
    manifest = raw_manifest()
    selection = build_selection(manifest, byte_budget=10, free_bytes=100, reserve_bytes=5,
                                filesystem=tmp_path)
    assert selection["source_key"] == source_key(source_identity(manifest))
    assert selection["selection"]["selected_bytes"] == 10
    assert selection["selection"]["skipped_file_count"] == 1
    second = build_selection(manifest, byte_budget=10, free_bytes=100, reserve_bytes=5,
                             filesystem=tmp_path)
    assert selection["selection_id"] == second["selection_id"]
    with AdmissionRegistry(tmp_path / "registry.sqlite3") as registry:
        assert registry.register_selection(selection) == selection
        assert registry.register_selection(second) == selection
        changed = {**selection, "selected_files": [selection["selected_files"][0]],
                   "selection": {**selection["selection"], "selected_bytes": 6}}
        with pytest.raises(RuntimeError, match="different immutable selection"):
            registry.register_selection(changed)
        output = registry.register_selection_file(selection)
        assert json.loads(output.read_text())["selection_id"] == selection["selection_id"]


def test_cli_dry_run_and_admit_stream_local_parquet_without_downloading(tmp_path, capsys):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "snapshot"
    parquet = data_dir / "part" / "a.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"id": "one", "text": "first local admission document"},
        {"id": "two", "text": "second local admission document"},
    ]), parquet)
    item = {"path": "part/a.parquet", "bytes": parquet.stat().st_size,
            "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest()}
    manifest_path = tmp_path / "raw-manifest.json"
    manifest_path.write_text(json.dumps({**raw_manifest(files=[item]),
        "repo": "owner/local", "revision": "commit-1", "config": "sample"}))
    registry_path = tmp_path / "admission" / "registry.sqlite3"
    common = ["--manifest", str(manifest_path), "--registry", str(registry_path),
              "--byte-budget", str(item["bytes"]), "--reserve-bytes", "0",
              "--filesystem", str(tmp_path)]

    main(["dry-run", *common, "--data-dir", str(data_dir)])
    preview = json.loads(capsys.readouterr().out)
    assert preview["local_scan"]["records"] == 2
    assert not registry_path.exists()

    main(["admit", *common, "--data-dir", str(data_dir)])
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["document_counters"]["registered"] == 2
    assert Path(admitted["selection_manifest"]).exists()
    with AdmissionRegistry(registry_path) as registry:
        assert registry.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
