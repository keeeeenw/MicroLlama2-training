"""Freeze deterministic held-out document prefixes for reproducible diagnostics."""
import hashlib
import json
from pathlib import Path

from ..data_io.corpus import documents, split_for, content_hash, SPLIT_SEED, VALIDATION_PERMILLE
from ..data_io.download import sha256
from ..data_io.transfer import atomic_json


def fixed_fixture(root, settings, directory):
    source = json.loads((root / "raw/fineweb-edu-10bt/manifest.json").read_text())
    tokenizer = json.loads((root / "tokenizer/source_manifest.json").read_text())
    shards = sorted(source["files"], key=lambda f: (f["bytes"], f["path"]))
    source_identity = [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
                       for item in shards]
    identity = {"source_revision": source["revision"], "source_shards": source_identity,
                "tokenizer_sha256": tokenizer["tokenizer_sha256"],
                "split_seed": SPLIT_SEED, "validation_permille": VALIDATION_PERMILLE,
                "documents": settings["documents"],
                "max_characters_per_document": settings["max_characters_per_document"]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fixture_path = directory / f"fixture-{key[:16]}.json"
    if fixture_path.exists():
        fixture = json.loads(fixture_path.read_text())
        if fixture.get("identity_sha256") != key or fixture.get("identity") != identity:
            raise RuntimeError(f"Cached evaluation fixture identity mismatch: {fixture_path}")
        if len(fixture.get("samples", [])) != settings["documents"]:
            raise RuntimeError(f"Cached evaluation fixture has the wrong sample count: {fixture_path}")
        return fixture, fixture_path
    location = Path(source.get("data_dir", root / "raw/fineweb-edu-10bt"))
    # Smallest source first gives the same fixture for cache and external workflows.
    for item in shards:
        path = location / item["path"]
        if not path.exists() or path.stat().st_size != item["bytes"]:
            continue
        if sha256(path) != item["sha256"]:
            raise RuntimeError("Evaluation source checksum mismatch")
        selected = []
        seen = set()
        for doc in documents(path):
            digest = content_hash(doc["text"])
            if split_for(doc["text"]) != "validation" or digest in seen:
                continue
            seen.add(digest)
            text = doc["text"][:settings["max_characters_per_document"]]
            selected.append({"id": doc["id"], "text": text, "content_hash": digest,
                             "truncated": len(text) < len(doc["text"])})
            if len(selected) == settings["documents"]:
                break
        if len(selected) != settings["documents"]:
            continue
        fixture = {"identity": identity, "identity_sha256": key,
                   "source": item["path"], "source_sha256": item["sha256"], "samples": selected}
        atomic_json(fixture_path, fixture)
        return fixture, fixture_path
    raise RuntimeError("No verified source with held-out documents is ready")

