import json

import numpy as np
import pytest

from microllama.data_io.blocks import GrowingTokenBlocks, DataWaitStopped
from microllama.data_io.prepare import prepare_source, update_summary
from microllama.data_io.corpus import split_for
from test_data import write_shard


def summary(path, complete=False, **extra):
    (path / "manifest.json").write_text(json.dumps({"source_revision": "fixture",
        "tokenizer_sha256": "fixture", "complete": complete, **extra}))


def test_append_resume_and_no_repetition_before_complete(tmp_path):
    summary(tmp_path)
    write_shard(tmp_path / "a", range(25))
    data = GrowingTokenBlocks(tmp_path, 4)
    data.next_batch(2)
    state = data.state_dict()
    write_shard(tmp_path / "b", range(100, 125))
    expected = data.next_batch(10)
    restored = GrowingTokenBlocks(tmp_path, 4)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(expected, restored.next_batch(10))
    assert restored.epoch == 0
    with pytest.raises(DataWaitStopped):
        restored.next_batch(1, lambda: True)
    summary(tmp_path, complete=True)
    restored.next_batch(1)
    assert restored.epoch == 1


def test_producer_failure_and_missing_chunk(tmp_path):
    summary(tmp_path)
    write_shard(tmp_path / "a", range(25))
    data = GrowingTokenBlocks(tmp_path, 4)
    state = data.state_dict()
    (tmp_path / "a" / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="removed"):
        data.load_state_dict(state)
    summary(tmp_path, status="failed", error="fixture failure")
    with pytest.raises(RuntimeError, match="fixture failure"):
        GrowingTokenBlocks(tmp_path, 4)


def test_chunked_preparation_resume_has_each_document_once(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers
    from microllama.data_io.encoding import (ENCODING_POLICY_VERSION,
        load_ordinary_text_tokenizer)
    raw_tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "fixture-tokenizer.json"
    raw_tok.save(str(tokenizer_path))
    tok = load_ordinary_text_tokenizer(tokenizer_path)
    val = next(f"hello validation {i}" for i in range(10000)
               if split_for(f"hello validation {i}") == "validation")
    docs = [{"id": str(i), "text": val if i % 4 == 0 else f"hello document {i}"}
            for i in range(40)]
    path = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pylist(docs), path)
    out = tmp_path / "tokens"
    out.mkdir()
    item = {"path": "a.parquet", "sha256": "fixture"}
    source = {"revision": "fixture"}
    assert prepare_source(path, item, source, tok, "fixture", 7, out, 20) == 40
    before = {p.parent.name: p.read_bytes() for p in out.glob("*/manifest.json")}
    assert len(before) > 1
    assert prepare_source(path, item, source, tok, "fixture", 7, out, 20) == 40
    assert before == {p.parent.name: p.read_bytes() for p in out.glob("*/manifest.json")}
    found = []
    for p in out.glob("*/manifest.json"):
        meta = json.loads(p.read_text())
        assert meta["encoding_policy_version"] == ENCODING_POLICY_VERSION
        for split in ("train", "validation"):
            values = np.fromfile(p.parent / f"{split}.bin", dtype="<u2")
            for line in (p.parent / f"{split}.documents.jsonl").read_text().splitlines():
                entry = json.loads(line)
                found.append(entry["id"])
                assert values[entry["offset"] + entry["tokens"] - 1] == 7
                assert split_for(docs[int(entry["id"])]["text"]) == split
            assert len(values) == meta["splits"][split]["tokens"]
    assert sorted(found, key=int) == [str(i) for i in range(40)]
    update_summary(out, source, "fixture", True)
    prepared_summary = json.loads((out / "manifest.json").read_text())
    assert prepared_summary["complete"]
    assert prepared_summary["encoding_policy_version"] == ENCODING_POLICY_VERSION


def test_legacy_chunks_remain_readable_but_cannot_be_extended(tmp_path):
    from microllama.data_io.blocks import TokenBlocks

    chunk = tmp_path / "a-000000"
    chunk.mkdir()
    values = np.asarray([1, 2, 3, 4, 5, 6], dtype="<u2")
    values.tofile(chunk / "train.bin")
    values.tofile(chunk / "validation.bin")
    legacy = {"source_revision": "fixture", "tokenizer_sha256": "fixture", "dtype": "<u2",
        "chunk_index": 0, "source_rows_end": 1,
        "splits": {name: {"tokens": 6, "documents": 1, "utf8_bytes": 5}
                   for name in ("train", "validation")}}
    manifest_path = chunk / "manifest.json"
    manifest_path.write_text(json.dumps(legacy))

    # The data reader stays compatible with published pre-policy chunks.
    assert TokenBlocks(tmp_path, 2).next_batch(1).shape == (1, 3)

    before = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="legacy/unspecified"):
        prepare_source(tmp_path / "unused.parquet", {"path": "a.parquet", "sha256": "fixture"},
                       {"revision": "fixture"}, None, "fixture", 7, tmp_path, 20)
    assert manifest_path.read_bytes() == before


def test_archive_publishes_verified_copy_and_rejects_corruption(tmp_path):
    import hashlib
    from microllama.data_io.archive_tokens import archive_chunk
    source = tmp_path / "source"
    write_shard(source, range(25))
    meta = json.loads((source / "manifest.json").read_text())
    for split in ("train", "validation"):
        meta["splits"][split]["sha256"] = hashlib.sha256((source / f"{split}.bin").read_bytes()).hexdigest()
    (source / "manifest.json").write_text(json.dumps(meta))
    destination = tmp_path / "archive"
    archive_chunk(source, destination)
    archive_chunk(source, destination)
    assert (source / "train.bin").read_bytes() == (destination / "train.bin").read_bytes()
    assert not destination.with_name("archive.partial").exists()
    (destination / "train.bin").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        archive_chunk(source, destination)


def test_prepared_only_rejects_incomplete_or_wrong_tokenizer(tmp_path):
    from scripts.pipeline.run_pipeline import validate_prepared_data
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"tokenizer_sha256": "fixture"}))
    summary(tmp_path, complete=False, train_tokens=100)
    with pytest.raises(RuntimeError, match="complete token corpus"):
        validate_prepared_data(tmp_path, tokenizer)
    summary(tmp_path, complete=True, train_tokens=100)
    assert validate_prepared_data(tmp_path, tokenizer)["train_tokens"] == 100
    tokenizer.write_text(json.dumps({"tokenizer_sha256": "different"}))
    with pytest.raises(RuntimeError, match="tokenizer mismatch"):
        validate_prepared_data(tmp_path, tokenizer)
