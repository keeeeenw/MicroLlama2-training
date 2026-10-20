"""Publish immutable token chunks from verified source shards as they arrive."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .corpus import documents, split_for, content_hash
from .download import sha256
from .encoding import (ENCODING_POLICY_VERSION, encode_ordinary_text_batch,
                       load_ordinary_text_tokenizer, require_encoding_policy)
from ..common.paths import configure_storage
from .transfer import atomic_json


def prepare_source(path, item, source, tokenizer, tokenizer_hash, eos, output, chunk_tokens):
    prefix = Path(item["path"]).stem
    existing = sorted(p for p in output.glob(f"{prefix}-*/manifest.json")
                      if not p.parent.name.endswith(".partial"))
    previous = [json.loads(p.read_text()) for p in existing]
    for i, meta in enumerate(previous):
        require_encoding_policy(meta, existing[i].parent)
        if (meta["source_revision"] != source["revision"] or meta["tokenizer_sha256"] != tokenizer_hash
                or meta["source_sha256"] != item["sha256"]
                or meta["chunk_index"] != i):
            raise RuntimeError("Incompatible published streaming chunks")
    skip_rows = previous[-1]["source_rows_end"] if previous else 0
    row = 0
    index = len(previous)
    handles = None
    tmp = None
    meta = None
    totals = None
    hashers = None

    def begin():
        nonlocal handles, tmp, meta, totals, hashers
        tmp = output / f"{prefix}-{index:06d}.partial"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        meta = {"source": Path(item["path"]).name, "source_revision": source["revision"],
                "source_sha256": item["sha256"], "tokenizer_sha256": tokenizer_hash,
                "encoding_policy_version": ENCODING_POLICY_VERSION,
                "dtype": "<u2", "chunk_index": index, "source_rows_start": row}
        handles = {s: ((tmp / f"{s}.bin").open("wb"),
                       (tmp / f"{s}.documents.jsonl").open("w")) for s in ("train", "validation")}
        totals = {s: {"tokens": 0, "documents": 0, "utf8_bytes": 0} for s in handles}
        hashers = {s: hashlib.sha256() for s in handles}

    def publish():
        nonlocal handles, index
        for pair in handles.values():
            for handle in pair:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        for split in totals:
            totals[split]["sha256"] = hashers[split].hexdigest()
        meta.update(source_rows_end=row, splits=totals)
        atomic_json(tmp / "manifest.json", meta)
        final = output / f"{prefix}-{index:06d}"
        tmp.rename(final)
        handles = None
        index += 1
        print(f"Published {final.name}: {totals['train']['tokens']:,} train tokens, "
              f"{totals['validation']['tokens']:,} validation tokens", flush=True)
        update_summary(output, source, tokenizer_hash, complete=False)

    batch = []

    def write_batch(batch):
        nonlocal row
        encodings = encode_ordinary_text_batch(tokenizer, [d["text"] for d in batch])
        for doc, encoding in zip(batch, encodings):
            if handles is None:
                begin()
            text = doc["text"]
            split = split_for(text)
            ids = encoding.ids + [eos]
            if max(ids) > 65535:
                raise RuntimeError("Token ID exceeds uint16")
            encoded = np.asarray(ids, dtype="<u2").tobytes()
            data, docs = handles[split]
            data.write(encoded)
            hashers[split].update(encoded)
            t = totals[split]
            size = len(text.encode("utf-8"))
            docs.write(json.dumps({"id": doc["id"], "content_hash": content_hash(text),
                                   "offset": t["tokens"], "tokens": len(ids), "utf8_bytes": size}) + "\n")
            t["tokens"] += len(ids)
            t["documents"] += 1
            t["utf8_bytes"] += size
            row += 1
            if totals["train"]["tokens"] >= chunk_tokens:
                publish()

    try:
        for doc in documents(path):
            if row < skip_rows:
                row += 1
                continue
            batch.append(doc)
            if len(batch) == 256:
                write_batch(batch)
                batch.clear()
        if batch:
            write_batch(batch)
        if handles is not None:
            publish()
    finally:
        if handles is not None:
            for pair in handles.values():
                for handle in pair:
                    handle.close()
    return row


def update_summary(output, source, digest, complete, **extra):
    paths = sorted(p for p in output.glob("*/manifest.json") if not p.parent.name.endswith(".partial"))
    metadata = [json.loads(p.read_text()) for p in paths]
    for path, manifest in zip(paths, metadata):
        require_encoding_policy(manifest, path.parent)
    summary = {"mode": "streaming_chunks", "source_revision": source["revision"],
               "tokenizer_sha256": digest, "encoding_policy_version": ENCODING_POLICY_VERSION,
               "complete": complete,
               "status": "complete" if complete else "preparing", "shards": [p.parent.name for p in paths],
               "train_tokens": sum(m["splits"]["train"]["tokens"] for m in metadata),
               "validation_tokens": sum(m["splits"]["validation"]["tokens"] for m in metadata)}
    summary.update(extra)
    atomic_json(output / "manifest.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="Prepared chunks; defaults to the data root's tokens directory")
    p.add_argument("--chunk-tokens", type=int, default=8_388_608)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.chunk_tokens < 1:
        p.error("Chunk token limit must be positive")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    root = configure_storage()
    args.output = args.output or root / "tokens"
    raw = root / "raw" / "fineweb-edu-10bt"
    source = json.loads((raw / "manifest.json").read_text())
    tok_path = root / "tokenizer" / "tokenizer.json"
    digest = sha256(tok_path)
    info = json.loads((root / "tokenizer" / "source_manifest.json").read_text())
    if digest != info["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer provenance mismatch")
    tok = load_ordinary_text_tokenizer(tok_path)
    args.output.mkdir(parents=True, exist_ok=True)
    for manifest_path in sorted(p for p in args.output.glob("*/manifest.json")
                                if not p.parent.name.endswith(".partial")):
        require_encoding_policy(json.loads(manifest_path.read_text()), manifest_path.parent)
    lock = (args.output / ".producer.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another streaming token producer is running")
    completed_dir = args.output / ".sources"
    completed_dir.mkdir(exist_ok=True)
    for marker in sorted(completed_dir.glob("*.json")):
        require_encoding_policy(json.loads(marker.read_text()), marker)
    update_summary(args.output, source, digest, complete=False)
    pending = []
    for item in source["files"]:
        marker = completed_dir / (Path(item["path"]).stem + ".json")
        if marker.exists():
            state = json.loads(marker.read_text())
            if state["revision"] != source["revision"] or state["sha256"] != item["sha256"]:
                raise RuntimeError("Completed source provenance mismatch")
            require_encoding_policy(state, marker)
            prefix = Path(item["path"]).stem
            manifests = sorted(args.output.glob(f"{prefix}-*/manifest.json"))
            if not manifests or json.loads(manifests[-1].read_text())["source_rows_end"] != state["documents"]:
                raise RuntimeError("Completed source chunks are missing")
        else:
            pending.append(item)
    try:
        while pending:
            # Download may publish its cache location while this producer waits.
            location = json.loads((raw / "manifest.json").read_text())
            directory = Path(location.get("data_dir", raw))
            ready = []
            for item in pending:
                final = directory / item["path"]
                if final.exists() and final.stat().st_size == item["bytes"]:
                    ready.append((item, final))
            if not ready:
                print(f"Waiting for a verified source shard ({len(pending)} remaining)", flush=True)
                time.sleep(10)
                continue
            ready.sort(key=lambda pair: pair[0]["path"])
            item, path = ready[0]
            if sha256(path) != item["sha256"]:
                raise RuntimeError(f"Source checksum mismatch: {path}")
            print(f"Tokenizing {path} into progressive chunks", flush=True)
            rows = prepare_source(path, item, source, tok, digest, info["eos_token_id"],
                                  args.output, args.chunk_tokens)
            atomic_json(completed_dir / (Path(item["path"]).stem + ".json"),
                        {"revision": source["revision"], "sha256": item["sha256"], "documents": rows,
                         "encoding_policy_version": ENCODING_POLICY_VERSION})
            pending.remove(item)
        update_summary(args.output, source, digest, complete=True)
        print("All source documents tokenized. The growing corpus is complete.", flush=True)
    except Exception as error:
        update_summary(args.output, source, digest, complete=False, status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
