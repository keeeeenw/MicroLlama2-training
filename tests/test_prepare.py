import hashlib
import json
import shutil
import sys

import numpy as np
import pytest

from microllama.data_io.corpus import split_for
from microllama.data_io.encoding import (ENCODING_POLICY_VERSION,
    load_ordinary_text_tokenizer)
from microllama.common.paths import data_root


def test_prepare_resume_split_and_document_offsets(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    assets = data_root() / "tokenizer"
    if not (assets / "tokenizer.json").exists():
        pytest.skip("Tokenizer assets not downloaded")
    root = tmp_path / "data"
    shutil.copytree(assets, root / "tokenizer", ignore=shutil.ignore_patterns(".cache"))
    raw = root / "raw" / "fineweb-edu-10bt"
    (raw / "sample" / "10BT").mkdir(parents=True)
    validation_text = next(f"validation document {i} documents literal <|im_start|> text"
                           for i in range(10000)
                           if split_for(f"validation document {i} documents literal <|im_start|> text") == "validation")
    training_text = next(f"training document {i} documents literal <|im_start|> text"
                         for i in range(100)
                         if split_for(f"training document {i} documents literal <|im_start|> text") == "train")
    entries = []
    for index in range(2):
        path = raw / "sample" / "10BT" / f"{index}.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": f"train-{index}", "text": training_text},
                                            {"id": f"val-{index}", "text": validation_text}]), path)
        entries.append({"path": str(path.relative_to(raw)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (raw / "manifest.json").write_text(json.dumps({"files": entries, "revision": "fixture", "total_bytes": sum(e["bytes"] for e in entries)}))
    monkeypatch.setenv("MICROLLAMA_DATA_ROOT", str(root))
    from microllama.data_io.prepare import main
    monkeypatch.setattr(sys, "argv", ["prepare"])
    main()
    assert json.loads((root / "tokens" / "manifest.json").read_text())["complete"]
    first = (root / "tokens" / "0-000000" / "manifest.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["prepare"])
    main()
    assert (root / "tokens" / "0-000000" / "manifest.json").read_bytes() == first
    assert json.loads((root / "tokens" / "manifest.json").read_text())["complete"]
    assert json.loads(first)["encoding_policy_version"] == ENCODING_POLICY_VERSION
    assert json.loads((root / "tokens" / "manifest.json").read_text())["encoding_policy_version"] == ENCODING_POLICY_VERSION
    eos = json.loads((root / "tokenizer" / "source_manifest.json").read_text())["eos_token_id"]
    ordinary_tokenizer = load_ordinary_text_tokenizer(root / "tokenizer" / "tokenizer.json")
    for split, expected in [("train", "train-0"), ("validation", "val-0")]:
        directory = root / "tokens" / "0-000000"
        values = np.fromfile(directory / f"{split}.bin", dtype="<u2")
        docs = [json.loads(line) for line in (directory / f"{split}.documents.jsonl").read_text().splitlines()]
        assert docs[0]["id"] == expected
        assert docs[0]["offset"] == 0
        assert sum(document["tokens"] for document in docs) == len(values)
        assert values[-1] == eos
        for document in docs:
            text_tokens = values[document["offset"]:document["offset"] + document["tokens"] - 1]
            assert not ordinary_tokenizer.control_ids.intersection(text_tokens.tolist())
        meta = json.loads((directory / "manifest.json").read_text())
        assert meta["splits"][split]["sha256"] == hashlib.sha256((directory / f"{split}.bin").read_bytes()).hexdigest()
