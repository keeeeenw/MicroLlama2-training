"""Bounded CPU preparation of an isolated, science-filtered Cosmopedia v2 pool."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

from .download import sha256
from .science_filter import BenchmarkOverlapBlocker, RULE_DIGEST, RULE_VERSION, classify_cosmopedia_science

REPO = "HuggingFaceTB/smollm-corpus"
CONFIG = "cosmopedia-v2"
PILOT_CAP = 5_000_000_000
BATCH_CAP = 30_000_000_000
RESERVE_BYTES = 100 * 2**30
HEADROOM_BYTES = 512 * 2**20
SEQUENCE_LENGTH = 8192
APPROVAL_SCOPE = "cosmopedia-science-preparation-v1"
PREPARATION_VERSION = "cosmopedia-science-prepare-v2"
SAMPLES_PER_CATEGORY = 16
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SOURCE_PATH = re.compile(r"cosmopedia-v2/train-(\d{5})-of-00104\.parquet")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Persist state before acquisition/publication, including directory metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require_owned_tree(root):
    """Refuse output links before any operation can reach another data tree."""
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Output contains a symlink: {path}")
    database = root / "admission.sqlite3"
    if database.exists() and database.stat().st_nlink != 1:
        raise ValueError("Owned admission database must not be hardlinked")


def frozen_identity(path):
    stat = Path(path).stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def require_sha(path, expected):
    if not isinstance(expected, str) or not _HEX64.fullmatch(expected) or sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def load_source_manifest(path, expected_sha):
    require_sha(path, expected_sha)
    source = read_json(path)
    if source.get("repo") != REPO or source.get("config") != CONFIG or not _HEX40.fullmatch(source.get("revision", "")):
        raise ValueError("An explicitly pinned official Cosmopedia v2 manifest is required")
    files = source.get("files", [])
    if len(files) != 104:
        raise ValueError("Source manifest must list all 104 official Parquet shards")
    numbers = set()
    for item in files:
        match = _SOURCE_PATH.fullmatch(item.get("path", ""))
        if not match or type(item.get("bytes")) is not int or item["bytes"] <= 0 or not _HEX64.fullmatch(item.get("sha256", "")):
            raise ValueError("Source manifest contains an invalid path, byte count or checksum")
        numbers.add(int(match.group(1)))
    if numbers != set(range(104)):
        raise ValueError("Source manifest contains repeated or missing official shards")
    return {**source, "files": sorted(files, key=lambda item: item["path"])}


def require_space(root, additional_bytes):
    needed = RESERVE_BYTES + HEADROOM_BYTES + max(0, int(additional_bytes))
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(f"Storage reserve would be crossed: free={free}, required={needed}")
    return free


class ReadOnlyAdmission:
    """Query the current registry without importing, copying or modifying it."""

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.identity = frozen_identity(self.path)
        self.assert_unchanged()
        # Root must verify the producer has exited before using this frozen mode.
        # immutable=1 avoids even creating SQLite WAL/SHM coordination files.
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=60)
        self.db.execute("PRAGMA query_only=ON")
        for table, required in (("exact", {"hash", "split"}), ("near", {"hash", "sig", "b0", "b1", "b2", "b3"})):
            columns = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            if not required.issubset(columns):
                self.db.close()
                raise ValueError(f"Current admission database is missing {table} provenance")
        self.assert_unchanged()

    def assert_unchanged(self):
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists() or sidecar.is_symlink():
                raise ValueError("Frozen admission reader requires absent WAL/SHM/journal and an exited producer")
        if frozen_identity(self.path) != self.identity:
            raise ValueError("Frozen admission database identity changed")

    def overlap(self, doc):
        if self.db.execute("SELECT 1 FROM exact WHERE hash=?", (doc["content_hash"],)).fetchone():
            return "legacy_exact_overlap"
        signature = doc["signature"]
        if signature is not None:
            bands = [int(signature, 16) >> (32 * i) & ((1 << 32) - 1) for i in range(4)]
            rows = self.db.execute("SELECT sig FROM near WHERE b0=? OR b1=? OR b2=? OR b3=? LIMIT 257", bands).fetchall()
            if len(rows) > 256:
                return "legacy_ambiguous_near_bucket"
            if any((int(signature, 16) ^ int(row[0], 16)).bit_count() <= 3 for row in rows):
                return "legacy_near_overlap"
        return None

    def close(self):
        self.db.close()


@dataclass(frozen=True)
class Options:
    output_root: Path
    source_manifest: Path
    source_manifest_sha256: str
    benchmark_documents: Path
    benchmark_sha256: str
    legacy_admission_db: Path
    tokenizer_dir: Path
    raw_cap_bytes: int
    target_valid_targets: int
    stage: str = "pilot"
    pilot_cap_bytes: int = PILOT_CAP
    batch_cap_bytes: int = BATCH_CAP
    cap_approval: Path | None = None
    cap_approval_sha256: str | None = None
    tokenizer_threads: int = 2
    chunk_tokens: int = 8_388_608
    prior_selection_state: Path | None = None
    prior_selection_sha256: str | None = None
    samples_per_category: int = SAMPLES_PER_CATEGORY


def _validate_options(options):
    if type(options.raw_cap_bytes) is not int or options.raw_cap_bytes <= 0 or options.target_valid_targets <= 0:
        raise ValueError("Explicit positive raw cap and valid-target goal are required")
    if not 0 < options.pilot_cap_bytes <= PILOT_CAP or not 0 < options.batch_cap_bytes <= BATCH_CAP:
        raise ValueError("Pilot and later batches cannot exceed 5 GB and 30 GB")
    if options.stage not in ("pilot", "continue") or not 1 <= options.tokenizer_threads <= 4 or options.chunk_tokens < 1:
        raise ValueError("Invalid preparation stage, CPU thread count or chunk limit")
    if not 16 <= options.samples_per_category <= 64:
        raise ValueError("Review samples must be between 16 and 64 per category per shard")
    if (options.prior_selection_state is None) != (options.prior_selection_sha256 is None):
        raise ValueError("Prior selection state and its checksum must be supplied together")
    root = options.output_root.resolve()
    legacy = options.legacy_admission_db.resolve(strict=True)
    tokenizer_dir = options.tokenizer_dir.resolve(strict=True)
    # Never place output in the existing growth tree, or place that tree in output.
    for original in (legacy.parent, tokenizer_dir):
        if root == original or root.is_relative_to(original) or original.is_relative_to(root):
            raise ValueError("Preparation root must be isolated from current growth/tokenizer data")
    return root, legacy, tokenizer_dir


def prior_acquisition(options, root, source):
    """Quarantine a frozen earlier pilot while charging its raw acquisitions."""
    if options.prior_selection_state is None:
        return {"skipped_shards": 0, "prior_raw_bytes": 0}
    path = options.prior_selection_state.resolve(strict=True)
    if root == path.parent or root.is_relative_to(path.parent) or path.parent.is_relative_to(root):
        raise ValueError("Replacement pool must be isolated from the prior science pool")
    require_sha(path, options.prior_selection_sha256)
    state = read_json(path)
    if state.get("phase") != "paused" or state.get("error"):
        raise ValueError("Prior pilot must be a frozen, unambiguous bounded stop")
    pin = state["pin"]
    if any(pin.get(key) != source[key] for key in ("repo", "revision", "config")) or pin["source_manifest_sha256"] != options.source_manifest_sha256:
        raise ValueError("Prior acquisitions use a different source pin")
    selections = state["selected"]
    if not selections or len(selections) >= len(source["files"]):
        raise ValueError("Prior acquisitions must be a nonempty proper inventory prefix")
    for number, item in enumerate(selections, 1):
        expected = source["files"][number - 1]
        if item["sequence"] != number or any(item[key] != expected[key] for key in ("path", "bytes", "sha256")):
            raise ValueError("Prior selected ledger differs from the pinned inventory")
    used = sum(item["bytes"] for item in selections)
    if state["raw_bytes_selected"] != used or state["raw_cap_bytes"] != options.raw_cap_bytes or used >= options.raw_cap_bytes:
        raise ValueError("Prior raw accounting or global cap differs")
    return {"skipped_shards": len(selections), "prior_raw_bytes": used,
            "prior_selection_state": str(path), "prior_selection_sha256": options.prior_selection_sha256,
            "prior_pool_quarantined": True}


def _approval(options, root, old_cap):
    if options.cap_approval is None or options.cap_approval_sha256 is None:
        raise ValueError("Separate checksummed approval record required to continue or increase the raw cap")
    require_sha(options.cap_approval, options.cap_approval_sha256)
    value = read_json(options.cap_approval)
    expected = {"scope": APPROVAL_SCOPE, "output_root": str(root),
                "source_manifest_sha256": options.source_manifest_sha256,
                "previous_raw_cap_bytes": old_cap, "raw_cap_bytes": options.raw_cap_bytes}
    if any(value.get(key) != wanted for key, wanted in expected.items()) or not value.get("authorization"):
        raise ValueError("Approval scope, output root, source pin or raw cap does not match")
    if options.prior_selection_state is not None:
        if value.get("prior_selection_state") != str(options.prior_selection_state.resolve()) or value.get("prior_selection_sha256") != options.prior_selection_sha256:
            raise ValueError("Approval must bind the prior acquisition ledger")
    return {**expected, "approval_sha256": options.cap_approval_sha256, "authorization": value["authorization"]}


def select_next(source, state, options):
    """Select whole shards without crossing cumulative, pilot or batch budgets."""
    selected = state["selected"]
    offset = state["pin"].get("raw_accounting", {}).get("skipped_shards", 0)
    prior = state["pin"].get("raw_accounting", {}).get("prior_raw_bytes", 0)
    if offset + len(selected) == len(source["files"]):
        return None, "inventory_exhausted"
    item = source["files"][offset + len(selected)]
    used = sum(entry["bytes"] for entry in selected)
    # The new pilot has its own 5GB stage bound, within the SAME global cap.
    limit = min(options.raw_cap_bytes - prior, options.pilot_cap_bytes) if options.stage == "pilot" else options.raw_cap_bytes - prior
    if used + item["bytes"] > limit:
        return None, "pilot_raw_cap" if options.stage == "pilot" else "raw_cap_exhausted"
    batch = 0 if options.stage == "pilot" else max(1, max((entry["batch"] for entry in selected), default=0))
    in_batch = sum(entry["bytes"] for entry in selected if entry["batch"] == batch)
    if in_batch + item["bytes"] > options.batch_cap_bytes:
        batch += 1
    if item["bytes"] > options.batch_cap_bytes:
        return None, "shard_exceeds_batch_cap"
    return {**item, "sequence": len(selected) + 1, "batch": batch}, None


def fetch_file(source, item, directory):
    """Acquire one selected shard into owned local staging, without a shared cache."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import constants
    if not constants.HF_HUB_DISABLE_XET:
        raise RuntimeError("Set HF_HUB_DISABLE_XET=1 before starting the preparation process")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(source["repo"], item["path"], repo_type="dataset",
                               revision=source["revision"], local_dir=directory, token=False))


def _sample(samples, row_id, record, reason, decision, limit=SAMPLES_PER_CATEGORY):
    """Keep fixed-size samples by stable hash; do not retain unbounded raw prose."""
    category = "accepted" if reason == "accepted_science" else "rejected"
    score = int(hashlib.sha256((category + ":" + row_id).encode()).hexdigest(), 16)
    heap = samples[category]
    if len(heap) == limit and -score <= heap[0][0]:
        return
    text, prompt = record.get("text"), record.get("prompt")
    excerpt_limit = 12000 if category == 'accepted' else 2000
    entry = {"id": row_id, "reason": reason, "domains": decision.domains,
             "format": record.get("format"), "audience": record.get("audience"),
             "text_excerpt": text[:excerpt_limit] if isinstance(text, str) else None,
             "text_excerpt_truncated": isinstance(text, str) and len(text) > excerpt_limit,
             "prompt_excerpt": prompt[:1000] if isinstance(prompt, str) else None,
             "sample_hash": f"{score:064x}"}
    candidate = (-score, row_id, entry)
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    else:
        heapq.heapreplace(heap, candidate)


def _raw_rows(path):
    import pyarrow.parquet as pq
    required = ("prompt", "text", "audience", "format", "seed_data")
    parquet = pq.ParquetFile(path)
    if not set(required).issubset(parquet.schema_arrow.names):
        raise ValueError("Selected Parquet does not have the Cosmopedia v2 schema")
    for batch in parquet.iter_batches(batch_size=256, columns=list(required), use_threads=False):
        yield from batch.to_pylist()


def filter_shard(root, source, item, raw, index, legacy, blocker, sample_limit=SAMPLES_PER_CATEGORY):
    """Commit own dedup admissions and filtered-file checksums in one SQL transaction."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .growth_prepare import annotate

    prefix = f"cs{item['sequence']:06d}"
    stage = root / "staging" / prefix
    filtered = stage / (prefix + ".parquet")
    decisions = stage / "accepted.decisions.jsonl"
    final_decisions = root / "receipts" / (prefix + ".decisions.jsonl")
    sample_path = stage / "pilot-samples.json"
    final_samples = root / "receipts" / (prefix + ".samples.json")
    key = "filter:" + prefix
    previous = index.receipt(key)
    if previous:
        if previous["original_sha256"] != item["sha256"]:
            raise ValueError("Filtered receipt source changed")
        require_sha(filtered, previous["filtered_sha256"])
        require_sha(decisions if decisions.exists() else final_decisions, previous["decisions_sha256"])
        require_sha(sample_path if sample_path.exists() else final_samples, previous["samples_sha256"])
        return filtered, previous
    partial = filtered.with_suffix(".parquet.partial")
    schema = pa.schema([("text", pa.string()), ("id", pa.string())])
    writer = pq.ParquetWriter(partial, schema)
    started = time.monotonic()
    counts, domains = Counter(), Counter()
    samples = {"accepted": [], "rejected": []}
    accepted, row_count, text_bytes = [], 0, 0
    index.db.execute("BEGIN IMMEDIATE")
    try:
        legacy.assert_unchanged()
        with decisions.open("w") as sidecar:
            for row_number, record in enumerate(_raw_rows(raw)):
                row_count += 1
                if row_number % 256 == 0:
                    require_space(root, HEADROOM_BYTES)
                # Rejected subjects never enter training, so avoid scanning their
                # text against the full benchmark index. Every retained row is scanned.
                decision = classify_cosmopedia_science(record)
                reason = decision.reason
                if decision.accepted and blocker.match(prompt=record["prompt"], text=record["text"]) is not None:
                    reason = "benchmark_overlap"
                if decision.accepted and reason != "benchmark_overlap":
                    # Upstream v2 has no unique ID; bind identity to pinned path and row.
                    row_id = f"{item['path']}:{row_number}"
                    doc = annotate([{"id": row_id, "text": record["text"]}])[0]
                    reason = legacy.overlap(doc)
                    if reason is None:
                        own_reason = index.admit(source["repo"], doc)
                        reason = "accepted_science" if own_reason == "accepted" else "science_" + own_reason
                    if reason == "accepted_science":
                        size = len(doc["text"].encode("utf-8"))
                        require_space(root, 4 * size + 4096)
                        accepted.append({"text": doc["text"], "id": row_id})
                        text_bytes += size
                        domains.update(decision.domains)
                        sidecar.write(json.dumps({"id": row_id, "row_index": row_number,
                            "content_hash": doc["content_hash"], "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                            "domains": decision.domains, "evidence": decision.evidence,
                            "audience": record.get("audience"), "format": record.get("format"),
                            "seed_data": record.get("seed_data")}, sort_keys=True) + "\n")
                counts[reason] += 1
                _sample(samples, f"{item['path']}:{row_number}", record, reason, decision, sample_limit)
                if len(accepted) == 256:
                    require_space(root, sum(len(doc["text"].encode("utf-8")) for doc in accepted) * 4)
                    writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
                    accepted.clear()
            if accepted:
                writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
            sidecar.flush()
            os.fsync(sidecar.fileno())
        writer.close()
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        partial.replace(filtered)
        sync_directory(stage)
        sync_directory(root / "staging")
        write_json(sample_path, {category: [entry[2] for entry in sorted(heap, key=lambda entry: -entry[0])]
                                 for category, heap in samples.items()})
        receipt = {"original_sha256": item["sha256"], "filtered_sha256": sha256(filtered),
                   "decisions_sha256": sha256(decisions), "samples_sha256": sha256(sample_path), "source_rows": row_count,
                   "accepted_utf8_bytes": text_bytes, "counts": dict(counts), "domains": dict(domains),
                   "filter_rule_version": RULE_VERSION, "filter_rule_digest": RULE_DIGEST,
                   "benchmark_digest": blocker.digest, "filter_wall_seconds": time.monotonic() - started}
        legacy.assert_unchanged()
        index.save(key, receipt)
        index.db.commit()
        return filtered, receipt
    except BaseException:
        writer.close()
        index.db.rollback()
        raise


def _packing_bytes(filtered, tokenizer, eos):
    """Measure actual ordinary token IDs before allocating their output payload."""
    from .corpus import documents
    from .encoding import encode_ordinary_text_batch
    total, batch = 0, []
    for doc in documents(filtered):
        batch.append(doc)
        if len(batch) == 256:
            total += sum(2 * (len(encoding.ids) + 1) + len(doc["id"].encode()) + 512
                         for doc, encoding in zip(batch, encode_ordinary_text_batch(tokenizer, [d["text"] for d in batch])))
            batch.clear()
    if batch:
        total += sum(2 * (len(encoding.ids) + 1) + len(doc["id"].encode()) + 512
                     for doc, encoding in zip(batch, encode_ordinary_text_batch(tokenizer, [d["text"] for d in batch])))
    return total


def pack_shard(root, source, item, filtered, receipt, index, tokenizer, tokenizer_info, pin, options, legacy,
               processing_started, filter_reused):
    from .prepare import prepare_source
    from .growth_prepare import DEDUP_POLICY
    from .growth_reader import usable_targets

    prefix = f"cs{item['sequence']:06d}"
    stage = root / "staging" / prefix
    output = stage / "tokens"
    # A crash may have moved a prefix of these chunks into immutable publication.
    # Rebuild owned staging from the committed filtered source; compare any finals.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(exist_ok=True)
    require_space(root, _packing_bytes(filtered, tokenizer, tokenizer_info["eos_token_id"]))
    prepare_source(filtered, {"path": filtered.name, "sha256": receipt["filtered_sha256"]}, source,
                   tokenizer, tokenizer_info["tokenizer_sha256"], tokenizer_info["eos_token_id"], output, options.chunk_tokens)
    final_decisions = root / "receipts" / (prefix + ".decisions.jsonl")
    if final_decisions.exists():
        require_sha(final_decisions, receipt["decisions_sha256"])
    else:
        (stage / "accepted.decisions.jsonl").replace(final_decisions)
    final_samples = root / "receipts" / (prefix + ".samples.json")
    if final_samples.exists():
        require_sha(final_samples, receipt["samples_sha256"])
    else:
        (stage / "pilot-samples.json").replace(final_samples)
    sync_directory(stage)
    sync_directory(root / "receipts")
    chunks = {}
    for file in sorted(output.glob("*/manifest.json")):
        if file.parent.name.endswith(".partial"):
            continue
        manifest = read_json(file)
        manifest.update(source_repo=source["repo"], source_config=CONFIG, original_file=item["path"],
            original_source_sha256=item["sha256"], selection_sequence=item["sequence"], sequence_length=SEQUENCE_LENGTH,
            preparation_pin=pin, filter_rule_version=RULE_VERSION, filter_rule_digest=RULE_DIGEST,
            benchmark_digest=pin["benchmark_digest"], benchmark_documents_sha256=pin["benchmark_documents_sha256"],
            dedup_policy=DEDUP_POLICY, accepted_decisions={"path": str(final_decisions), "sha256": receipt["decisions_sha256"]},
            pilot_samples={"path": str(final_samples), "sha256": receipt["samples_sha256"]},
            document_sidecars={split: sha256(file.parent / f"{split}.documents.jsonl") for split in ("train", "validation")})
        manifest["valid_targets"] = usable_targets(file.parent / "train.bin", SEQUENCE_LENGTH,
                                                   tokenizer_info["eos_token_id"], tokenizer_info["pad_token_id"])
        write_json(file, manifest)
        sync_directory(file.parent)
        final = root / "tokens" / file.parent.name
        if final.exists():
            if sha256(final / "manifest.json") != sha256(file):
                raise ValueError("Previously published science chunk changed")
            shutil.rmtree(file.parent)
        else:
            file.parent.rename(final)
        chunks[final.name] = {"manifest_sha256": sha256(final / "manifest.json"), "valid_targets": manifest["valid_targets"]}
        sync_directory(output)
        sync_directory(root / "tokens")
    published = {**receipt, "chunks": chunks, "valid_targets": sum(c["valid_targets"] for c in chunks.values()),
                 "processing_wall_seconds": time.monotonic() - processing_started,
                 "timing_scope": "this-attempt", "filter_reused": filter_reused}
    legacy.assert_unchanged()
    write_json(root / "receipts" / (prefix + ".json"), published)
    index.save("published:" + prefix, published)
    index.db.commit()
    return published


def verify_published(root, receipt, pin):
    from .encoding import require_encoding_policy
    from .growth_reader import usable_targets
    for name, entry in receipt["chunks"].items():
        if not re.fullmatch(r"cs\d{6}-\d{6}", name):
            raise ValueError("Invalid science chunk name")
        directory = root / "tokens" / name
        require_sha(directory / "manifest.json", entry["manifest_sha256"])
        manifest = read_json(directory / "manifest.json")
        require_encoding_policy(manifest, directory)
        if manifest["preparation_pin"] != pin or manifest["filter_rule_digest"] != RULE_DIGEST:
            raise ValueError("Published preparation identity differs")
        for split in ("train", "validation"):
            require_sha(directory / f"{split}.bin", manifest["splits"][split]["sha256"])
            if (directory / f"{split}.bin").stat().st_size != 2 * manifest["splits"][split]["tokens"]:
                raise ValueError("Published token length changed")
            require_sha(directory / f"{split}.documents.jsonl", manifest["document_sidecars"][split])
        require_sha(Path(manifest["accepted_decisions"]["path"]), manifest["accepted_decisions"]["sha256"])
        require_sha(Path(manifest["pilot_samples"]["path"]), manifest["pilot_samples"]["sha256"])
        actual = usable_targets(directory / "train.bin", SEQUENCE_LENGTH, pin["eos"], pin["pad"])
        if actual != entry["valid_targets"] or actual != manifest["valid_targets"]:
            raise ValueError("Published 8K valid-target count differs")


def _remove_stage(root, sequence):
    stage = root / "staging" / f"cs{sequence:06d}"
    if stage.is_symlink() or stage.resolve().parent != (root / "staging").resolve():
        raise ValueError("Refusing to delete unowned staging")
    if stage.exists():
        shutil.rmtree(stage)
        sync_directory(root / "staging")


def require_measured_pilot(root, state, pin):
    """A selected or failed download is not a completed, reviewable pilot."""
    database = root / "admission.sqlite3"
    if not database.exists():
        raise ValueError("A verified measured pilot is required before continuation")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        for item in state["selected"]:
            if item["batch"] != 0:
                continue
            prefix = f"cs{item['sequence']:06d}"
            row = connection.execute("SELECT value FROM receipts WHERE key=?", ("published:" + prefix,)).fetchone()
            if row is None:
                continue
            receipt = json.loads(row[0])
            if receipt["original_sha256"] != item["sha256"] or receipt["source_rows"] <= 0:
                raise ValueError("Pilot receipt source/row counts differ")
            require_sha(root / "receipts" / (prefix + ".decisions.jsonl"), receipt["decisions_sha256"])
            require_sha(root / "receipts" / (prefix + ".samples.json"), receipt["samples_sha256"])
            verify_published(root, receipt, pin)
            return
    finally:
        connection.close()
    raise ValueError("A verified measured pilot is required before continuation")


def run_preparation(options, *, fetch=fetch_file):
    """Run/resume one explicitly bounded stage. No existing growth state is written."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root, legacy_path, tokenizer_dir = _validate_options(options)
    source = load_source_manifest(options.source_manifest, options.source_manifest_sha256)
    accounting = prior_acquisition(options, root, source)
    require_sha(options.benchmark_documents, options.benchmark_sha256)
    benchmark = read_json(options.benchmark_documents)
    blocker = BenchmarkOverlapBlocker(benchmark["documents"] if isinstance(benchmark, dict) else benchmark)
    tokenizer_info = read_json(tokenizer_dir / "source_manifest.json")
    require_sha(tokenizer_dir / "tokenizer.json", tokenizer_info["tokenizer_sha256"])
    for name in ("eos_token_id", "pad_token_id"):
        if type(tokenizer_info.get(name)) is not int or not 0 <= tokenizer_info[name] <= 65535:
            raise ValueError("Pinned EOS and padding IDs required")
    pin = {"repo": source["repo"], "revision": source["revision"], "config": CONFIG,
           "source_manifest_sha256": options.source_manifest_sha256, "filter_rule_version": RULE_VERSION,
           "filter_rule_digest": RULE_DIGEST, "benchmark_documents_sha256": options.benchmark_sha256,
           "benchmark_digest": blocker.digest, "tokenizer_sha256": tokenizer_info["tokenizer_sha256"],
           "eos": tokenizer_info["eos_token_id"], "pad": tokenizer_info["pad_token_id"],
           "legacy_admission": {"path": str(legacy_path), "mode": "immutable-frozen", **frozen_identity(legacy_path)},
           "preparation_version": PREPARATION_VERSION, "samples_per_category_per_shard": options.samples_per_category,
           "raw_accounting": accounting,
           "sequence_length": SEQUENCE_LENGTH, "target_valid_targets": options.target_valid_targets,
           "pilot_cap_bytes": options.pilot_cap_bytes, "batch_cap_bytes": options.batch_cap_bytes}
    state_path = root / "prepare-state.json"
    owner_path = root / ".science-owner.json"
    if root.exists():
        require_owned_tree(root)
    if root.exists() and not owner_path.exists() and any(root.iterdir()):
        raise ValueError("Output root is nonempty and is not an owned science preparation")
    initial_approval = _approval(options, root, 0) if not state_path.exists() else None
    root.mkdir(parents=True, exist_ok=True)
    from contextlib import ExitStack
    with ExitStack() as stack:
        lock = stack.enter_context((root / ".prepare.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another science producer holds the output lock")
        if options.prior_selection_state is not None:
            # Exclude an accidental old-filter writer for the complete new run.
            old_lock = stack.enter_context((options.prior_selection_state.parent / ".prepare.lock").open("a"))
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_sha(options.prior_selection_state, options.prior_selection_sha256)
        if owner_path.exists():
            if read_json(owner_path) != {"schema_version": 1, "output_root": str(root), "pin": pin}:
                raise ValueError("Existing science preparation identity differs")
        else:
            write_json(owner_path, {"schema_version": 1, "output_root": str(root), "pin": pin})
        state = read_json(state_path) if state_path.exists() else {
            "schema_version": 1, "pin": pin, "raw_cap_bytes": options.raw_cap_bytes,
            "stage": "pilot", "selected": [], "approval_history": [initial_approval], "chunks": {}, "complete": False}
        if state["pin"] != pin:
            raise ValueError("Existing preparation pin changed")
        old_cap = state["raw_cap_bytes"]
        if options.raw_cap_bytes < old_cap:
            raise ValueError("Cannot silently lower an existing recorded raw budget")
        if options.raw_cap_bytes != old_cap or options.stage != state["stage"]:
            approval = _approval(options, root, old_cap)
            if state["stage"] == "continue" and options.stage == "pilot":
                raise ValueError("Cannot return a continuation ledger to pilot mode")
            if options.stage == "continue":
                require_measured_pilot(root, state, pin)
            state["approval_history"].append(approval)
            state.update(raw_cap_bytes=options.raw_cap_bytes, stage=options.stage)
        # Initial continue cannot bypass the measured pilot stage.
        if options.stage == "continue" and not state["selected"]:
            raise ValueError("Run a measured pilot before continuation")
        for number, item in enumerate(state["selected"], 1):
            expected = source["files"][accounting["skipped_shards"] + number - 1]
            if item["sequence"] != number or any(item[key] != expected[key] for key in ("path", "bytes", "sha256")):
                raise ValueError("Selected ledger differs from pinned inventory")
        if accounting["prior_raw_bytes"] + sum(item["bytes"] for item in state["selected"]) > options.raw_cap_bytes:
            raise ValueError("Selected ledger exceeds the explicit raw cap")
        # Reconstruct counted coverage from verified SQL publication receipts.
        state.update(chunks={}, complete=False)
        state.pop("error", None)
        state.pop("stop_reason", None)
        for child in ("staging", "tokens", "receipts"):
            (root / child).mkdir(exist_ok=True)
        write_json(state_path, state)
        from .encoding import load_ordinary_text_tokenizer
        from .growth_prepare import DedupIndex
        os.environ["RAYON_NUM_THREADS"] = str(options.tokenizer_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import pyarrow as pa
        pa.set_cpu_count(options.tokenizer_threads)
        pa.set_io_thread_count(options.tokenizer_threads)
        tokenizer = load_ordinary_text_tokenizer(tokenizer_dir / "tokenizer.json")
        legacy = ReadOnlyAdmission(legacy_path)
        index = DedupIndex(root / "admission.sqlite3")
        verified_sequences = set()
        try:
            started = time.monotonic()
            while True:
                legacy.assert_unchanged()
                # Complete interrupted selected shards before selecting any new file.
                pending = None
                for item in state["selected"]:
                    published = index.receipt(f"published:cs{item['sequence']:06d}")
                    if published is None:
                        pending = item
                        break
                    if item["sequence"] not in verified_sequences:
                        verify_published(root, published, pin)
                        verified_sequences.add(item["sequence"])
                    state["chunks"].update(published["chunks"])
                    _remove_stage(root, item["sequence"])
                total = sum(chunk["valid_targets"] for chunk in state["chunks"].values())
                if pending is None and total >= options.target_valid_targets:
                    state.update(complete=True, stop_reason="target_reached")
                    break
                if pending is None:
                    pending, reason = select_next(source, state, options)
                    if pending is None:
                        state.update(complete=False, stop_reason=reason)
                        break
                    # Reserve raw, filtered output, SQL/sidecars and packing headroom.
                    require_space(root, pending["bytes"] * 5)
                    state["selected"].append(pending)
                    state.update(phase="selected", updated_unix=time.time())
                    write_json(state_path, state)
                prefix = f"cs{pending['sequence']:06d}"
                shard_started = time.monotonic()
                stage = root / "staging" / prefix
                stage.mkdir(parents=True, exist_ok=True)
                raw_root = stage / "raw"
                raw_root.mkdir(exist_ok=True)
                raw = raw_root / pending["path"]
                filter_reused = index.receipt("filter:" + prefix) is not None
                if not filter_reused:
                    if not raw.exists():
                        require_space(root, pending["bytes"] * 5)
                        state.update(phase="acquiring", active_sequence=pending["sequence"], updated_unix=time.time())
                        write_json(state_path, state)
                        fetched = Path(fetch(source, pending, raw_root))
                        require_owned_tree(stage)
                        if fetched.resolve() != raw.resolve() or fetched.is_symlink():
                            raise ValueError("Acquisition returned a path outside owned staging")
                    if raw.stat().st_size != pending["bytes"]:
                        raise ValueError("Selected raw shard byte count differs")
                    require_sha(raw, pending["sha256"])
                    write_json(root / "receipts" / (prefix + ".raw.json"), {
                        "path": pending["path"], "bytes": pending["bytes"], "sha256": pending["sha256"],
                        "repo": source["repo"], "revision": source["revision"], "sequence": pending["sequence"]})
                filtered, receipt = filter_shard(root, source, pending, raw, index, legacy, blocker, options.samples_per_category)
                published = pack_shard(root, source, pending, filtered, receipt, index, tokenizer, tokenizer_info, pin,
                    options, legacy, shard_started, filter_reused)
                verify_published(root, published, pin)
                verified_sequences.add(pending["sequence"])
                state["chunks"].update(published["chunks"])
                state.update(phase="published", updated_unix=time.time())
                write_json(state_path, state)
                _remove_stage(root, pending["sequence"])
            state.update(phase="complete" if state["complete"] else "paused", updated_unix=time.time(),
                         raw_bytes_selected=sum(item["bytes"] for item in state["selected"]),
                         cumulative_raw_bytes_selected=accounting["prior_raw_bytes"] + sum(item["bytes"] for item in state["selected"]),
                         valid_targets=sum(chunk["valid_targets"] for chunk in state["chunks"].values()),
                         last_run_wall_seconds=time.monotonic() - started)
            legacy.assert_unchanged()
            if options.prior_selection_state is not None:
                require_sha(options.prior_selection_state, options.prior_selection_sha256)
            write_json(state_path, state)
            from .prepare import update_summary
            update_summary(root / "tokens", source, tokenizer_info["tokenizer_sha256"], state["complete"],
                status=state["phase"], stop_reason=state["stop_reason"], preparation_pin=pin,
                valid_targets=state["valid_targets"], raw_bytes_selected=state["raw_bytes_selected"],
                raw_cap_bytes=options.raw_cap_bytes)
            write_json(root / "tokens/manifest.json", read_json(root / "tokens/manifest.json"))
            return state
        except BaseException as error:
            state.update(phase="failed", complete=False, error=f"{type(error).__name__}: {error}", updated_unix=time.time())
            write_json(state_path, state)
            raise
        finally:
            index.db.close()
            legacy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-root", "source-manifest", "benchmark-documents", "legacy-admission-db", "tokenizer-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-manifest-sha256", "benchmark-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--raw-cap-bytes", type=int, required=True)
    parser.add_argument("--target-valid-targets", type=int, required=True)
    parser.add_argument("--stage", choices=("pilot", "continue"), default="pilot")
    parser.add_argument("--pilot-cap-bytes", type=int, default=PILOT_CAP)
    parser.add_argument("--batch-cap-bytes", type=int, default=BATCH_CAP)
    parser.add_argument("--cap-approval", type=Path)
    parser.add_argument("--cap-approval-sha256")
    parser.add_argument("--tokenizer-threads", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=8_388_608)
    parser.add_argument("--prior-selection-state", type=Path)
    parser.add_argument("--prior-selection-sha256")
    parser.add_argument("--samples-per-category", type=int, default=SAMPLES_PER_CATEGORY)
    args = parser.parse_args()
    state = run_preparation(Options(**vars(args)))
    print(json.dumps({key: state[key] for key in ("phase", "complete", "stop_reason", "raw_bytes_selected", "valid_targets")}))


if __name__ == "__main__":
    main()
