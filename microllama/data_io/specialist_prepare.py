"""Bounded, isolated admission of broad Cosmopedia v2 and FineMath 4+."""
from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time

from .download import sha256
from .encoding import ENCODING_POLICY_VERSION, load_ordinary_text_tokenizer, require_encoding_policy
from .growth_prepare import DEDUP_POLICY, DedupIndex, annotate
from .growth_reader import digest, usable_targets
from .prepare import prepare_source, update_summary
from .science_filter import BenchmarkOverlapBlocker
from .science_prepare import (ReadOnlyAdmission, _packing_bytes, fetch_file, read_json,
                             frozen_identity, require_owned_tree, require_sha, require_space, sync_directory, write_json)

APPROVAL_SCOPE = "user-approved-four-source-preparation-v1"
VERSION = "specialist-prepare-v1"
RAW_CAP = 125_000_000_000
PILOT_CAP = 5_000_000_000
BATCH_CAP = 30_000_000_000
SEQUENCE_LENGTH = 8192
POLICY = {"version": VERSION, "cosmopedia": "broad-generated-text-no-subject-filter",
          "finemath": "int_score>=4", "empty_or_malformed": "reject",
          "benchmark": "prompt-and-text-lexical-overlap", "dedup": DEDUP_POLICY,
          "validation": "content-hash-split-v1", "training_text": "text-only"}
POLICY_DIGEST = digest(POLICY)
SOURCE_SPECS = {
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 104),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus", 64),
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Options:
    output_root: Path
    source_manifest: Path
    source_manifest_sha256: str
    benchmark_documents: Path
    benchmark_sha256: str
    legacy_admission_db: Path
    tokenizer_dir: Path
    target_valid_targets: dict[str, int]
    cap_approval: Path
    cap_approval_sha256: str
    prior_ledgers: tuple[tuple[Path, str], ...] = ()
    raw_cap_bytes: int = RAW_CAP
    pilot_cap_bytes: int = PILOT_CAP
    batch_cap_bytes: int = BATCH_CAP
    stage: str = "pilot"
    tokenizer_threads: int = 2
    chunk_tokens: int = 8_388_608
    samples_per_category: int = 64
    admission_workers: int = 1
    bulk_authorization: Path | None = None
    bulk_authorization_sha256: str | None = None


def load_sources(path, expected_sha):
    require_sha(path, expected_sha)
    value = read_json(path)
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_SPECS):
        raise ValueError("Manifest must contain exactly the two official specialist sources")
    for name, (repo, config, count) in SOURCE_SPECS.items():
        source = sources[name]
        if source.get("repo") != repo or source.get("config") != config or not HEX40.fullmatch(source.get("revision", "")):
            raise ValueError("Specialist source repository, configuration or revision is invalid")
        files = source.get("files", [])
        pattern = re.compile(re.escape(config) + rf"/train-(\d{{5}})-of-{count:05d}\.parquet")
        numbers = set()
        for item in files:
            match = pattern.fullmatch(item.get("path", ""))
            if not match or type(item.get("bytes")) is not int or item["bytes"] <= 0 or not HEX64.fullmatch(item.get("sha256", "")):
                raise ValueError("Source shard has an invalid path, byte count or SHA")
            numbers.add(int(match.group(1)))
        if len(files) != count or numbers != set(range(count)):
            raise ValueError("All official specialist shards must be pinned exactly once")
        source["files"] = sorted(files, key=lambda item: item["path"])
    return sources


def acquisition_key(source, item):
    return (source["repo"], source["revision"], item["path"], item["sha256"])


def prior_acquisitions(options, sources):
    """Charge the unique prior source selections, never quarantine tokens as training."""
    unique, evidence = {}, []
    advertised = {acquisition_key(source, item): item for source in sources.values() for item in source["files"]}
    for path, expected_sha in options.prior_ledgers:
        path = Path(path).resolve(strict=True)
        require_sha(path, expected_sha)
        state = read_json(path)
        if state.get("phase") not in ("paused", "complete") or state.get("error"):
            raise ValueError("Prior producer must have a frozen unambiguous bounded stop")
        pin = state.get("pin", {})
        selections = state.get("selected")
        if not isinstance(selections, list) or not selections:
            raise ValueError("Prior acquisition ledger has no bound selections")
        local_keys = set()
        for item in selections:
            key = (pin.get("repo"), pin.get("revision"), item.get("path"), item.get("sha256"))
            expected = advertised.get(key)
            if expected is None or item.get("bytes") != expected["bytes"] or key in local_keys:
                raise ValueError("Prior source selection differs from the pinned official inventory")
            local_keys.add(key)
            unique[key] = expected["bytes"]
        selected_bytes = sum(item["bytes"] for item in selections)
        if "raw_bytes_selected" in state:
            if state["raw_bytes_selected"] != selected_bytes:
                raise ValueError("Prior selected-raw accounting does not reconcile")
        elif pin.get("preparation_version") not in ("cosmopedia-science-prepare-v1", "cosmopedia-science-prepare-v2"):
            raise ValueError("Prior ledger without aggregate has an unknown preparation schema")
        evidence.append({"path": str(path), "sha256": expected_sha, "selected_count": len(selections)})
    return unique, evidence


def validate_options(options):
    if ((options.bulk_authorization is None) != (options.bulk_authorization_sha256 is None) or
            (options.bulk_authorization is not None and options.stage != "continue")):
        raise ValueError("Direct bulk authorization needs its checksum and the continuation stage")
    if set(options.target_valid_targets) != set(SOURCE_SPECS) or any(
            type(v) is not int or v < 1 for v in options.target_valid_targets.values()):
        raise ValueError("Positive valid-target goals required for both specialist sources")
    if any(type(v) is not int for v in (options.raw_cap_bytes, options.pilot_cap_bytes, options.batch_cap_bytes)) or (
            not 0 < options.raw_cap_bytes <= RAW_CAP or not 0 < options.pilot_cap_bytes <= PILOT_CAP or not 0 < options.batch_cap_bytes <= BATCH_CAP):
        raise ValueError("Specialist acquisition cannot exceed the 125GB/5GB/30GB bounds")
    if (options.stage not in ("pilot", "continue") or type(options.admission_workers) is not int or
            not 1 <= options.admission_workers <= 32 or type(options.tokenizer_threads) is not int or
            not 1 <= options.tokenizer_threads <= 16 or options.chunk_tokens < 1):
        raise ValueError("Invalid bounded stage or CPU packing settings")
    if not 16 <= options.samples_per_category <= 64:
        raise ValueError("Review sample count must be between 16 and 64")
    root = options.output_root.resolve()
    for original in (options.legacy_admission_db.resolve(strict=True).parent,
                     options.tokenizer_dir.resolve(strict=True),
                     *(Path(p).resolve(strict=True).parent for p, _ in options.prior_ledgers)):
        if root == original or root.is_relative_to(original) or original.is_relative_to(root):
            raise ValueError("New preparation must be isolated from existing roots")
    return root


def require_approval(options, root, prior_evidence):
    require_sha(options.cap_approval, options.cap_approval_sha256)
    approval = read_json(options.cap_approval)
    expected = {"scope": APPROVAL_SCOPE, "output_root": str(root),
                "source_manifest_sha256": options.source_manifest_sha256,
                "benchmark_documents_sha256": options.benchmark_sha256,
                "raw_cap_bytes": options.raw_cap_bytes,
                "target_valid_targets": options.target_valid_targets,
                "prior_ledgers": [{"path": p["path"], "sha256": p["sha256"]} for p in prior_evidence]}
    if any(approval.get(k) != v for k, v in expected.items()) or not approval.get("authorization"):
        raise ValueError("Checksummed user approval does not bind this preparation and its bounds")
    return {**expected, "approval_sha256": options.cap_approval_sha256, "authorization": approval["authorization"]}


def require_direct_bulk_authorization(options, root, state_path, pin):
    """Allow an explicitly approved review-after-preparation workflow, without claiming review passed."""
    if options.bulk_authorization is None or options.bulk_authorization_sha256 is None:
        raise ValueError("Checksummed direct bulk authorization required")
    require_sha(options.bulk_authorization, options.bulk_authorization_sha256)
    value = read_json(options.bulk_authorization)
    expected = {"scope": "user-approved-direct-specialist-bulk-v1", "output_root": str(root),
                "prepare_state_sha256": sha256(state_path), "preparation_pin_digest": digest(pin),
                "review_timing": "after-full-preparation-before-training"}
    if any(value.get(k) != v for k, v in expected.items()) or not value.get("authorization"):
        raise ValueError("Direct bulk authorization differs from the stopped state or preparation bounds")
    return value


def select_next(sources, state, options, prior):
    used = {acquisition_key(sources[item["source"]], item) for item in state["selected"]}
    totals = {name: sum(chunk["valid_targets"] for chunk in state["chunks"][name].values()) for name in sources}
    pending = [name for name in sources if totals[name] < options.target_valid_targets[name]]
    if not pending:
        return None, "target_reached"
    selected_bytes = sum(item["bytes"] for item in state["selected"])
    prior_bytes = sum(prior.values())
    batch = 0 if options.stage == "pilot" else max(1, max((i["batch"] for i in state["selected"]), default=0))
    in_batch = sum(i["bytes"] for i in state["selected"] if i["batch"] == batch)
    candidates, reasons = [], []
    represented = {item["source"] for item in state["selected"]}
    for name in sorted(pending, key=lambda n: (n in represented, totals[n] / options.target_valid_targets[n], n)):
        files = [item for item in sources[name]["files"] if acquisition_key(sources[name], item) not in used | set(prior)]
        if not files:
            reasons.append("inventory_exhausted")
            continue
        item = files[0]
        if prior_bytes + selected_bytes + item["bytes"] > options.raw_cap_bytes:
            reasons.append("raw_cap_exhausted")
            continue
        if options.stage == "pilot" and selected_bytes + item["bytes"] > options.pilot_cap_bytes:
            reasons.append("pilot_raw_cap")
            continue
        if item["bytes"] > options.batch_cap_bytes:
            reasons.append("shard_exceeds_batch_cap")
            continue
        next_batch = batch + 1 if in_batch + item["bytes"] > options.batch_cap_bytes else batch
        candidates.append({**item, "source": name, "sequence": len(state["selected"]) + 1, "batch": next_batch})
    return (candidates[0], None) if candidates else (None, "pilot_raw_cap" if "pilot_raw_cap" in reasons else reasons[0])


def rows(path, name):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    required = {"text", "prompt", "audience", "format", "seed_data"} if name == "cosmopedia" else {"text", "int_score"}
    if not required.issubset(file.schema_arrow.names):
        raise ValueError("Official specialist Parquet schema differs from the bound adapter")
    optional = {"id", "url", "domain"}.intersection(file.schema_arrow.names)
    for batch in file.iter_batches(batch_size=256, columns=sorted(required | optional), use_threads=False):
        yield from batch.to_pylist()


def quality_reason(record, name):
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        return "empty_or_malformed_text"
    if name == "cosmopedia" and not isinstance(record.get("prompt"), str):
        return "malformed_prompt"
    if name == "finemath" and (type(record.get("int_score")) is not int or record["int_score"] < 4):
        return "score_below_four_or_malformed"
    return None


def screen_record(record, source_name, row_id, blocker, legacy):
    """Read-only checks may run in parallel; new-source admission remains ordered."""
    reason = quality_reason(record, source_name)
    if reason is None and blocker.match(prompt=record.get("prompt", ""), text=record["text"]) is not None:
        reason = "benchmark_overlap"
    doc = None
    if reason is None:
        doc = annotate([{"id": row_id, "text": record["text"]}])[0]
        reason = legacy.overlap(doc)
    return record, row_id, reason, doc


def initialize_screen_worker(benchmark_path, benchmark_sha256, legacy_path, legacy_identity):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require_sha(Path(benchmark_path), benchmark_sha256)
    benchmark = read_json(Path(benchmark_path))
    global _worker_blocker, _worker_legacy
    _worker_blocker = BenchmarkOverlapBlocker(benchmark["documents"] if isinstance(benchmark, dict) else benchmark)
    _worker_legacy = ReadOnlyAdmission(Path(legacy_path))
    if _worker_legacy.identity != legacy_identity:
        raise ValueError("Parallel legacy reader differs from the frozen parent identity")


def screen_batch(task):
    source_name, source_path, start, records = task
    _worker_legacy.assert_unchanged()
    return [screen_record(record, source_name, f"{source_name}:{source_path}:{start + number}",
                          _worker_blocker, _worker_legacy) for number, record in enumerate(records)]


def screened_rows(raw, item, legacy, blocker, pool=None):
    if pool is None:
        for number, record in enumerate(rows(raw, item["source"])):
            row_id = f"{item['source']}:{item['path']}:{number}"
            yield screen_record(record, item["source"], row_id, blocker, legacy)
        return
    pending, batch, start = deque(), [], 0
    for record in rows(raw, item["source"]):
        batch.append(record)
        if len(batch) == 256:
            pending.append(pool.submit(screen_batch, (item["source"], item["path"], start, batch)))
            start += len(batch); batch = []
            # Bound memory and consume in source order regardless of worker completion order.
            if len(pending) >= pool._max_workers * 2:
                yield from pending.popleft().result()
    if batch:
        pending.append(pool.submit(screen_batch, (item["source"], item["path"], start, batch)))
    for future in pending:
        yield from future.result()


def filter_shard(root, source, item, raw, index, legacy, blocker, sample_limit, pool=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    prefix = f"sp{item['sequence']:06d}"
    stage = root / "staging" / prefix
    filtered = stage / (prefix + ".parquet")
    decisions = stage / "decisions.jsonl"
    samples_path = stage / "samples.json"
    schema = pa.schema([("text", pa.string()), ("id", pa.string())])
    writer = pq.ParquetWriter(filtered.with_suffix(".partial"), schema)
    accepted, counts, samples = [], Counter(), {"accepted": [], "rejected": []}
    source_rows = 0
    index.db.execute("BEGIN IMMEDIATE")
    try:
        with decisions.open("w") as sidecar:
            for number, (record, row_id, reason, doc) in enumerate(screened_rows(raw, item, legacy, blocker, pool)):
                source_rows += 1
                if number % 256 == 0:
                    require_space(root, 512 * 2**20)
                    legacy.assert_unchanged()
                if reason is None:
                    reason = index.admit(source["repo"], doc)
                counts[reason] += 1
                category = "accepted" if reason == "accepted" else "rejected"
                sample = {"id": row_id, "reason": reason,
                          "text_excerpt": record.get("text", "")[:12000] if isinstance(record.get("text"), str) else None,
                          "prompt_excerpt": record.get("prompt", "")[:1000] if isinstance(record.get("prompt"), str) else None,
                          "int_score": record.get("int_score"), "format": record.get("format"),
                          "audience": record.get("audience"), "seed_data": record.get("seed_data")}
                sample["sample_hash"] = hashlib.sha256(row_id.encode()).hexdigest()
                samples[category].append(sample)
                samples[category].sort(key=lambda s: s["sample_hash"])
                del samples[category][sample_limit:]
                if reason == "accepted":
                    accepted.append({"text": doc["text"], "id": row_id})
                    sidecar.write(json.dumps({"id": row_id, "row_index": number, "content_hash": doc["content_hash"],
                        "prompt_sha256": hashlib.sha256(record.get("prompt", "").encode()).hexdigest(),
                        "upstream_id": record.get("id"), "int_score": record.get("int_score"),
                        "format": record.get("format"), "audience": record.get("audience"),
                        "seed_data": record.get("seed_data")}, sort_keys=True) + "\n")
                if len(accepted) == 256:
                    require_space(root, sum(len(d["text"].encode()) for d in accepted) * 4)
                    writer.write_table(pa.Table.from_pylist(accepted, schema=schema)); accepted.clear()
            if accepted:
                require_space(root, sum(len(d["text"].encode()) for d in accepted) * 4)
                writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
            sidecar.flush(); os.fsync(sidecar.fileno())
        writer.close()
        partial = filtered.with_suffix(".partial")
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        partial.replace(filtered)
        write_json(samples_path, samples)
        receipt = {"source": item["source"], "source_rows": source_rows, "counts": dict(counts),
                   "selection_sequence": item["sequence"], "original_file": item["path"],
                   "filtered_sha256": sha256(filtered), "decisions_sha256": sha256(decisions),
                   "samples_sha256": sha256(samples_path), "original_sha256": item["sha256"],
                   "policy_digest": POLICY_DIGEST, "benchmark_digest": blocker.digest}
        index.save("filter:" + prefix, receipt)
        legacy.assert_unchanged()
        index.db.commit()
        sync_directory(stage)
        return filtered, receipt
    except BaseException:
        writer.close(); index.db.rollback(); raise


def verify_published(root, source_name, receipt, pin):
    number = receipt.get("selection_sequence")
    if type(number) is not int or number < 1 or receipt.get("source") != source_name or (
            receipt.get("policy_digest") != POLICY_DIGEST or receipt.get("benchmark_digest") != pin["benchmark_digest"]):
        raise ValueError("Specialist admission receipt identity differs")
    prefix = f"sp{number:06d}"
    source = pin["sources"][source_name]
    originals = {item["path"]: item["sha256"] for item in source["files"]}
    if originals.get(receipt.get("original_file")) != receipt.get("original_sha256"):
        raise ValueError("Specialist receipt does not bind the pinned original shard")
    if type(receipt.get("source_rows")) is not int or receipt["source_rows"] < 0 or (
            any(type(n) is not int or n < 0 for n in receipt.get("counts", {}).values()) or
            sum(receipt.get("counts", {}).values()) != receipt["source_rows"]):
        raise ValueError("Specialist admission counts do not reconcile")
    decisions = root / "receipts" / (prefix + ".decisions.jsonl")
    samples = root / "receipts" / (prefix + ".samples.json")
    require_sha(decisions, receipt["decisions_sha256"])
    require_sha(samples, receipt["samples_sha256"])
    total = 0
    for name, entry in receipt["chunks"].items():
        if not re.fullmatch(re.escape(prefix) + r"-\d{6}", name):
            raise ValueError("Invalid specialist token chunk name")
        folder = root / "sources" / source_name / "tokens" / name
        require_sha(folder / "manifest.json", entry["manifest_sha256"])
        meta = read_json(folder / "manifest.json")
        require_encoding_policy(meta, folder)
        if meta.get("preparation_pin") != pin or meta.get("admission_policy_digest") != POLICY_DIGEST:
            raise ValueError("Specialist publication identity changed")
        if any(meta.get(k) != v for k, v in {
                "source_name": source_name, "source_repo": source["repo"], "source_config": source["config"],
                "source_revision": source["revision"], "original_file": receipt["original_file"],
                "original_source_sha256": receipt["original_sha256"], "selection_sequence": number,
                "tokenizer_sha256": pin["tokenizer_sha256"], "sequence_length": SEQUENCE_LENGTH,
                "benchmark_digest": pin["benchmark_digest"], "benchmark_documents_sha256": pin["benchmark_documents_sha256"],
                "dedup_policy": DEDUP_POLICY,
                "accepted_decisions": {"path": str(decisions), "sha256": receipt["decisions_sha256"]},
                "pilot_samples": {"path": str(samples), "sha256": receipt["samples_sha256"]}}.items()):
            raise ValueError("Specialist chunk source or audit sidecar identity differs")
        for split in ("train", "validation"):
            path = folder / (split + ".bin")
            if path.stat().st_size != meta["splits"][split]["tokens"] * 2:
                raise ValueError("Specialist payload size mismatch")
            require_sha(path, meta["splits"][split]["sha256"])
            require_sha(folder / (split + ".documents.jsonl"), meta["document_sidecars"][split])
        if type(entry.get("valid_targets")) is not int or entry["valid_targets"] < 0 or (
                usable_targets(folder / "train.bin", SEQUENCE_LENGTH, pin["eos"], pin["pad"]) != entry["valid_targets"] or
                meta.get("valid_targets") != entry["valid_targets"]):
            raise ValueError("Specialist valid-target receipt differs")
        total += entry["valid_targets"]
    if receipt.get("valid_targets") != total:
        raise ValueError("Specialist per-shard valid-target aggregate differs")


def pack_shard(root, source, item, filtered, receipt, tokenizer, tokenizer_info, pin, options):
    prefix = f"sp{item['sequence']:06d}"
    stage = root / "staging" / prefix
    output = stage / "tokens"; output.mkdir()
    require_space(root, _packing_bytes(filtered, tokenizer, tokenizer_info["eos_token_id"]))
    prepare_source(filtered, {"path": filtered.name, "sha256": receipt["filtered_sha256"]}, source,
                   tokenizer, tokenizer_info["tokenizer_sha256"], tokenizer_info["eos_token_id"], output, options.chunk_tokens)
    final_receipts = root / "receipts"
    for filename, label in (("decisions.jsonl", "decisions"), ("samples.json", "samples")):
        target = final_receipts / (prefix + "." + filename)
        (stage / filename).replace(target)
        require_sha(target, receipt[label + "_sha256"])
    sync_directory(final_receipts)
    destination = root / "sources" / item["source"] / "tokens"
    destination.mkdir(parents=True, exist_ok=True)
    chunks = {}
    for file in sorted(output.glob("*/manifest.json")):
        meta = read_json(file)
        meta.update(source_repo=source["repo"], source_config=source["config"], original_file=item["path"],
                    original_source_sha256=item["sha256"], selection_sequence=item["sequence"],
                    source_name=item["source"], sequence_length=SEQUENCE_LENGTH, preparation_pin=pin,
                    admission_policy=POLICY, admission_policy_digest=POLICY_DIGEST,
                    dedup_policy=DEDUP_POLICY, benchmark_digest=pin["benchmark_digest"],
                    benchmark_documents_sha256=pin["benchmark_documents_sha256"],
                    accepted_decisions={"path": str(final_receipts / (prefix + ".decisions.jsonl")), "sha256": receipt["decisions_sha256"]},
                    pilot_samples={"path": str(final_receipts / (prefix + ".samples.json")), "sha256": receipt["samples_sha256"]},
                    document_sidecars={s: sha256(file.parent / (s + ".documents.jsonl")) for s in ("train", "validation")})
        meta["valid_targets"] = usable_targets(file.parent / "train.bin", SEQUENCE_LENGTH, pin["eos"], pin["pad"])
        write_json(file, meta)
        final = destination / file.parent.name
        if final.exists():
            raise ValueError("Unexpected preexisting specialist publication")
        file.parent.rename(final)
        chunks[final.name] = {"manifest_sha256": sha256(final / "manifest.json"), "valid_targets": meta["valid_targets"]}
        sync_directory(final)
    sync_directory(destination)
    result = {**receipt, "chunks": chunks, "valid_targets": sum(c["valid_targets"] for c in chunks.values())}
    write_json(final_receipts / (prefix + ".json"), result)
    return result


def run_preparation(options, *, fetch=fetch_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["RAYON_NUM_THREADS"] = str(options.tokenizer_threads)
    os.environ["ARROW_NUM_THREADS"] = str(options.tokenizer_threads)
    root = validate_options(options)
    sources = load_sources(options.source_manifest, options.source_manifest_sha256)
    prior, prior_evidence = prior_acquisitions(options, sources)
    approval = require_approval(options, root, prior_evidence)
    if sum(prior.values()) >= options.raw_cap_bytes:
        raise ValueError("Prior unique acquisitions already exhaust the combined cap")
    require_sha(options.benchmark_documents, options.benchmark_sha256)
    benchmark = read_json(options.benchmark_documents)
    blocker = BenchmarkOverlapBlocker(benchmark["documents"] if isinstance(benchmark, dict) else benchmark)
    info = read_json(options.tokenizer_dir / "source_manifest.json")
    require_sha(options.tokenizer_dir / "tokenizer.json", info["tokenizer_sha256"])
    if info.get("eos_token_id") != 7 or info.get("pad_token_id") != 0:
        raise ValueError("The preserved native EOS7/pad0 encoding is required")
    pin = {"sources": sources, "source_manifest_sha256": options.source_manifest_sha256,
           "tokenizer_sha256": info["tokenizer_sha256"], "eos": 7, "pad": 0,
           "sequence_length": SEQUENCE_LENGTH, "benchmark_documents_sha256": options.benchmark_sha256,
           "benchmark_digest": blocker.digest, "admission_policy_digest": POLICY_DIGEST,
           "version": VERSION, "target_valid_targets": options.target_valid_targets,
           "raw_cap_bytes": options.raw_cap_bytes, "pilot_cap_bytes": options.pilot_cap_bytes,
           "batch_cap_bytes": options.batch_cap_bytes, "prior_ledgers": prior_evidence,
           "prior_unique_raw_bytes": sum(prior.values()), "approval_sha256": approval["approval_sha256"],
           "legacy_admission": {"path": str(options.legacy_admission_db.resolve()),
                                **frozen_identity(options.legacy_admission_db)},
           "encoding_policy_version": ENCODING_POLICY_VERSION}
    owner = {"schema_version": 1, "output_root": str(root), "pin": pin}
    if root.exists():
        require_owned_tree(root)
        if not (root / ".specialist-owner.json").exists() and any(root.iterdir()):
            raise ValueError("Nonempty root does not belong to this preparation")
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((root / ".prepare.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for parent in sorted({Path(p).resolve().parent for p, _ in options.prior_ledgers}):
            previous_lock = stack.enter_context((parent / ".prepare.lock").open("a"))
            fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Revalidate only after excluding the old producers.
        if prior_acquisitions(options, sources) != (prior, prior_evidence):
            raise ValueError("Prior acquisition ledger changed while taking locks")
        owner_path = root / ".specialist-owner.json"
        if owner_path.exists() and read_json(owner_path) != owner:
            raise ValueError("Existing specialist preparation identity differs")
        write_json(owner_path, owner)
        state_path = root / "prepare-state.json"
        existed = state_path.exists()
        state = read_json(state_path) if existed else {"schema_version": 1, "pin": pin, "phase": "paused",
            "stage": "pilot", "selected": [], "chunks": {s: {} for s in sources}, "complete": False,
            "raw_bytes_selected": 0, "cumulative_raw_bytes_selected": sum(prior.values())}
        if state.get("pin") != pin:
            raise ValueError("Existing preparation source, policy, target or budget differs")
        if state.get("phase") not in ("paused", "complete") or state.get("error"):
            raise RuntimeError("Prior preparation attempt is failed or uncertain; inspection required")
        if state["stage"] == "continue" and options.stage == "pilot":
            raise ValueError("A bulk continuation cannot return to pilot")
        selected_keys = set()
        expected_chunks = {s: {} for s in sources}
        for number, item in enumerate(state["selected"], 1):
            if item.get("source") not in sources or item.get("sequence") != number:
                raise ValueError("Selected ledger source or order differs")
            key = acquisition_key(sources[item["source"]], item)
            advertised = {acquisition_key(sources[item["source"]], f): f for f in sources[item["source"]]["files"]}
            if key not in advertised or item["bytes"] != advertised[key]["bytes"] or key in selected_keys or key in prior:
                raise ValueError("Selected specialist shard differs or repeats a prior acquisition")
            selected_keys.add(key)
            receipt = read_json(root / "receipts" / f"sp{number:06d}.json")
            if receipt["original_sha256"] != item["sha256"] or receipt["source"] != item["source"]:
                raise ValueError("Completed receipt source differs")
            verify_published(root, item["source"], receipt, pin)
            if any(state["chunks"][item["source"]].get(n) != c for n, c in receipt["chunks"].items()):
                raise ValueError("Published chunk ledger differs")
            expected_chunks[item["source"]].update(receipt["chunks"])
        if expected_chunks != state["chunks"] or state["raw_bytes_selected"] != sum(i["bytes"] for i in state["selected"]) or (
                state["cumulative_raw_bytes_selected"] != sum(prior.values()) + state["raw_bytes_selected"]):
            raise ValueError("Published chunk inventory or selected-raw totals differ")
        if state["cumulative_raw_bytes_selected"] > options.raw_cap_bytes:
            raise ValueError("Existing selected-raw totals exceed the combined cap")
        actual_targets = {s: sum(c["valid_targets"] for c in expected_chunks[s].values()) for s in sources}
        if state.get("valid_targets") is not None and state["valid_targets"] != actual_targets:
            raise ValueError("Saved valid-target totals differ from verified receipts")
        if state["complete"] and any(actual_targets[s] < options.target_valid_targets[s] for s in sources):
            raise ValueError("Completed specialist acquisition lacks its declared usable coverage")
        if state["selected"]:
            require_sha(root / "admission.sqlite3", state.get("admission_sha256"))
            for suffix in ("-wal", "-shm", "-journal"):
                if Path(str(root / "admission.sqlite3") + suffix).exists():
                    raise ValueError("Stopped specialist admission ledger has unresolved SQLite sidecars")
        if state["stage"] == "continue":
            if state.get("direct_bulk_authorization"):
                direct = state["direct_bulk_authorization"]
                require_sha(Path(direct["path"]), direct["sha256"])
            else:
                require_sha(root / "pilot-review.json", state.get("bulk_review_sha256"))
        if options.stage == "continue" and state["stage"] != "continue":
            if not state["selected"] or {i["source"] for i in state["selected"]} != set(sources):
                raise ValueError("Both sources need a measured bounded pilot before bulk continuation")
            if options.bulk_authorization is not None:
                require_direct_bulk_authorization(options, root, state_path, pin)
                state["direct_bulk_authorization"] = {"path": str(options.bulk_authorization.resolve()),
                    "sha256": options.bulk_authorization_sha256,
                    "review_timing": "after-full-preparation-before-training"}
            else:
                review_path = root / "pilot-review.json"
                review = read_json(review_path) if review_path.exists() else {}
                if review.get("phase") != "passed" or review.get("prepare_state_sha256") != sha256(state_path) or not review.get("manual_samples_reviewed"):
                    raise ValueError("Checksummed pilot state and manual review required before bulk continuation")
                if set(review.get("sources_reviewed", [])) != set(sources):
                    raise ValueError("Manual review must cover both admitted specialist sources")
                state["bulk_review_sha256"] = sha256(review_path)
        if options.stage == "continue" and not existed:
            raise ValueError("An initial bulk run cannot bypass pilot")
        state["stage"] = options.stage
        (root / "receipts").mkdir(exist_ok=True)
        (root / "staging").mkdir(exist_ok=True)
        write_json(state_path, state)
        if state["complete"]:
            return state
        index = DedupIndex(root / "admission.sqlite3")
        legacy = ReadOnlyAdmission(options.legacy_admission_db)
        tokenizer = load_ordinary_text_tokenizer(options.tokenizer_dir / "tokenizer.json")
        pool = None
        try:
            if options.admission_workers > 1:
                pool = stack.enter_context(ProcessPoolExecutor(options.admission_workers,
                    mp_context=multiprocessing.get_context("spawn"), initializer=initialize_screen_worker,
                    initargs=(str(options.benchmark_documents), options.benchmark_sha256,
                              str(options.legacy_admission_db), legacy.identity)))
            while True:
                item, reason = select_next(sources, state, options, prior)
                if item is None:
                    state.update(stop_reason=reason, complete=reason == "target_reached")
                    break
                source = sources[item["source"]]
                prefix = f"sp{item['sequence']:06d}"
                stage = root / "staging" / prefix
                stage.mkdir()
                raw_root = stage / "raw"; raw_root.mkdir()
                # Bound and persist the selected identity before any acquisition.
                state["selected"].append(item)
                state.update(phase="acquiring", active_sequence=item["sequence"], updated_unix=time.time(),
                    raw_bytes_selected=sum(i["bytes"] for i in state["selected"]),
                    cumulative_raw_bytes_selected=sum(prior.values()) + sum(i["bytes"] for i in state["selected"]))
                write_json(state_path, state)
                require_space(root, item["bytes"] * 5)
                raw = Path(fetch(source, item, raw_root))
                expected_path = raw_root / item["path"]
                require_owned_tree(stage)
                if raw.is_symlink() or raw.resolve() != expected_path.resolve() or raw.stat().st_size != item["bytes"]:
                    raise ValueError("Acquisition escaped owned staging or has wrong byte count")
                require_sha(raw, item["sha256"])
                write_json(root / "receipts" / (prefix + ".raw.json"), {**item, "repo": source["repo"], "revision": source["revision"]})
                state.update(phase="filtering", updated_unix=time.time()); write_json(state_path, state)
                filtered, receipt = filter_shard(root, source, item, raw, index, legacy, blocker,
                                                options.samples_per_category, pool)
                state.update(phase="packing", updated_unix=time.time()); write_json(state_path, state)
                published = pack_shard(root, source, item, filtered, receipt, tokenizer, info, pin, options)
                verify_published(root, item["source"], published, pin)
                state["chunks"][item["source"]].update(published["chunks"])
                state.update(phase="published", updated_unix=time.time()); write_json(state_path, state)
                legacy.assert_unchanged()
                shutil.rmtree(stage)
                sync_directory(root / "staging")
            state.update(phase="complete" if state["complete"] else "paused", updated_unix=time.time(),
                         valid_targets={s: sum(c["valid_targets"] for c in state["chunks"][s].values()) for s in sources})
            legacy.assert_unchanged()
            if prior_acquisitions(options, sources) != (prior, prior_evidence):
                raise ValueError("Frozen prior acquisition ledger changed")
            index.db.commit()
            index.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            state["admission_sha256"] = sha256(root / "admission.sqlite3")
            write_json(state_path, state)
            for name, source in sources.items():
                output = root / "sources" / name / "tokens"; output.mkdir(parents=True, exist_ok=True)
                update_summary(output, source, info["tokenizer_sha256"], state["valid_targets"][name] >= options.target_valid_targets[name],
                    preparation_pin=pin, valid_targets=state["valid_targets"][name], source_name=name, stop_reason=state["stop_reason"])
            return state
        except BaseException as error:
            state.update(phase="failed", complete=False, error=f"{type(error).__name__}: {error}", updated_unix=time.time())
            write_json(state_path, state)
            raise
        finally:
            index.db.close(); legacy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-root", "source-manifest", "benchmark-documents", "legacy-admission-db", "tokenizer-dir", "cap-approval"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-manifest-sha256", "benchmark-sha256", "cap-approval-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cosmopedia-target", type=int, required=True)
    parser.add_argument("--finemath-target", type=int, required=True)
    parser.add_argument("--prior-ledger", nargs=2, action="append", default=[], metavar=("PATH", "SHA256"))
    parser.add_argument("--stage", choices=("pilot", "continue"), default="pilot")
    parser.add_argument("--raw-cap-bytes", type=int, default=RAW_CAP)
    parser.add_argument("--pilot-cap-bytes", type=int, default=PILOT_CAP)
    parser.add_argument("--batch-cap-bytes", type=int, default=BATCH_CAP)
    parser.add_argument("--tokenizer-threads", type=int, default=2)
    parser.add_argument("--admission-workers", type=int, default=1,
                        help="Read-only screening workers; ordered SQLite admission stays in the parent")
    parser.add_argument("--bulk-authorization", type=Path,
                        help="Checksummed explicit user authorization to review after full preparation")
    parser.add_argument("--bulk-authorization-sha256")
    args = vars(parser.parse_args())
    args["target_valid_targets"] = {"cosmopedia": args.pop("cosmopedia_target"), "finemath": args.pop("finemath_target")}
    args["prior_ledgers"] = tuple((Path(p), h) for p, h in args.pop("prior_ledger"))
    run_preparation(Options(**args))


if __name__ == "__main__":
    main()
