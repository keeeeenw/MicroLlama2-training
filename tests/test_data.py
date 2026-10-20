import json

import numpy as np

from microllama.data_io.corpus import split_for
from microllama.data_io.blocks import TokenBlocks


def write_shard(path, values):
    path.mkdir()
    for split in ("train", "validation"):
        np.asarray(values, dtype="<u2").tofile(path / f"{split}.bin")
    (path / "manifest.json").write_text(json.dumps({
        "source_revision": "fixture", "tokenizer_sha256": "fixture",
        "splits": {s: {"tokens": len(values)} for s in ("train", "validation")}}))


def test_document_split_groups_whitespace_duplicates():
    assert split_for("One two\nthree") == split_for("One   two three")


def test_resume_across_shards_and_epochs(tmp_path):
    write_shard(tmp_path / "a", range(25))
    write_shard(tmp_path / "b", range(100, 125))
    first = TokenBlocks(tmp_path, 4)
    first.next_batch(11)
    state = first.state_dict()
    expected = first.next_batch(8)
    second = TokenBlocks(tmp_path, 4)
    second.load_state_dict(state)
    np.testing.assert_array_equal(expected, second.next_batch(8))
    assert second.epoch > 0


def test_all_blocks_once_per_epoch(tmp_path):
    write_shard(tmp_path / "a", range(25))
    data = TokenBlocks(tmp_path, 4)
    rows = data.next_batch(6)
    assert sorted(rows[:, 0]) == [0, 4, 8, 12, 16, 20]
    assert np.all(rows[:, 1:] == rows[:, :-1] + 1)


def test_changed_dataset_rejects_resume(tmp_path):
    import pytest
    write_shard(tmp_path / "a", range(25))
    state = TokenBlocks(tmp_path, 4).state_dict()
    write_shard(tmp_path / "b", range(100, 125))
    with pytest.raises(RuntimeError, match="fingerprint"):
        TokenBlocks(tmp_path, 4).load_state_dict(state)
