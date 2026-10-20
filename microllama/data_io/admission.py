"""Bounded provenance admission registry for future corpus expansion.

This module records source identity, byte-bounded file selections and normalized
document identity. It does not download or prepare data, and existing training
readers do not consult it yet.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile

from .corpus import content_hash, documents
from .download import sha256
from ..common.paths import data_root


SCHEMA_VERSION = 1
DEFAULT_SPLIT_SEED = 42
DEFAULT_VALIDATION_PERMILLE = 5


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_identity(manifest):
    """Return the stable, pinned identity fields from a raw-source manifest."""
    repo = manifest.get("repo")
    revision = manifest.get("revision")
    config = manifest.get("config")
    if not isinstance(repo, str) or not repo or not isinstance(revision, str) or not revision:
        raise ValueError("Source manifest must pin non-empty repo and revision values")
    return {"repo": repo, "revision": revision, "config": config}


def source_key(identity):
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"source-{digest[:24]}"


def stable_source_doc_id(value):
    """Preserve the type as well as value when namespacing upstream IDs."""
    return canonical_json({"type": type(value).__name__, "value": value})


def content_split(normalized_hash, seed=DEFAULT_SPLIT_SEED,
                  validation_permille=DEFAULT_VALIDATION_PERMILLE):
    """Assign a duplicate group deterministically, using the existing split recipe."""
    if not 0 <= validation_permille <= 1000:
        raise ValueError("validation_permille must be between 0 and 1000")
    bucket = int(hashlib.blake2b(f"{seed}:{normalized_hash}".encode(), digest_size=8).hexdigest(), 16) % 1000
    return "validation" if bucket < validation_permille else "train"


def validate_raw_manifest(manifest):
    identity = source_identity(manifest)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Source manifest must contain a non-empty files list")
    total = 0
    seen = set()
    normalized_files = []
    for item in files:
        rel = PurePosixPath(str(item.get("path", "")))
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe source path in manifest: {item.get('path')!r}")
        if str(rel) in seen:
            raise ValueError(f"Duplicate source path in manifest: {rel}")
        seen.add(str(rel))
        size = item.get("bytes")
        digest = item.get("sha256")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid byte size for source file {rel}")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)):
            raise ValueError(f"Source file {rel} has no pinned SHA-256")
        normalized_files.append({"path": str(rel), "bytes": size, "sha256": digest.lower()})
        total += size
    advertised = manifest.get("total_bytes")
    if advertised is not None and advertised != total:
        raise ValueError(f"Manifest total_bytes {advertised} does not match file sum {total}")
    return identity, normalized_files, total


def select_files(files, byte_budget):
    """Select whole manifest files in order without exceeding a raw-byte cap."""
    if not isinstance(byte_budget, int) or byte_budget < 0:
        raise ValueError("byte_budget must be a non-negative integer")
    selected, skipped = [], []
    used = 0
    for item in files:
        if used + item["bytes"] <= byte_budget:
            selected.append(dict(item))
            used += item["bytes"]
        else:
            skipped.append({**item, "reason": "byte_budget"})
    return selected, skipped


def validate_storage_budget(selected_bytes, byte_budget, free_bytes, reserve_bytes):
    """Reject over-budget selections or selections that consume the free-space reserve."""
    values = (selected_bytes, byte_budget, free_bytes, reserve_bytes)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Byte counts and reserves must be non-negative integers")
    if selected_bytes > byte_budget:
        raise RuntimeError(f"Selected raw data is over byte budget: {selected_bytes} > {byte_budget}")
    required = selected_bytes + reserve_bytes
    if free_bytes < required:
        raise RuntimeError(f"Insufficient free space: need {required} bytes, have {free_bytes}")
    return {"selected_bytes": selected_bytes, "byte_budget": byte_budget,
            "free_bytes": free_bytes, "reserve_bytes": reserve_bytes,
            "required_free_bytes": required}


def build_selection(manifest, byte_budget, free_bytes, reserve_bytes, filesystem):
    identity, files, manifest_bytes = validate_raw_manifest(manifest)
    selected, skipped = select_files(files, byte_budget)
    if not selected:
        raise RuntimeError("No complete source file fits the byte budget")
    selected_bytes = sum(item["bytes"] for item in selected)
    resources = validate_storage_budget(selected_bytes, byte_budget, free_bytes, reserve_bytes)
    core = {"schema_version": SCHEMA_VERSION, "source_identity": identity,
            "source_key": source_key(identity), "manifest_sha256": hashlib.sha256(
                canonical_json(manifest).encode("utf-8")).hexdigest(),
            "byte_budget": byte_budget, "selected_files": selected}
    selection_id = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return {**core, "selection_id": selection_id,
            "selection": {"manifest_file_count": len(files),
                          "manifest_bytes": manifest_bytes,
                          "selected_file_count": len(selected),
                          "selected_bytes": selected_bytes,
                          "skipped_file_count": len(skipped),
                          "skipped_bytes": sum(item["bytes"] for item in skipped),
                          "skipped_files": skipped},
            "resource_check": {**resources, "filesystem": str(Path(filesystem).resolve())}}


class AdmissionRegistry:
    """SQLite document identity, duplicate-group split, and selection registry."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS registry_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_key TEXT PRIMARY KEY,
                identity_json TEXT NOT NULL,
                identity_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS content_groups (
                normalized_hash TEXT PRIMARY KEY,
                split TEXT NOT NULL CHECK (split IN ('train','validation','test')),
                first_source_key TEXT NOT NULL REFERENCES sources(source_key),
                first_doc_id TEXT NOT NULL,
                document_count INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS documents (
                source_key TEXT NOT NULL REFERENCES sources(source_key),
                source_doc_id TEXT NOT NULL,
                normalized_hash TEXT NOT NULL REFERENCES content_groups(normalized_hash),
                split TEXT NOT NULL CHECK (split IN ('train','validation','test')),
                utf8_bytes INTEGER NOT NULL,
                provenance_json TEXT NOT NULL,
                PRIMARY KEY (source_key, source_doc_id)
            );
            CREATE INDEX IF NOT EXISTS documents_by_hash ON documents(normalized_hash);
            CREATE INDEX IF NOT EXISTS documents_by_split ON documents(split);
            CREATE TABLE IF NOT EXISTS selections (
                selection_id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL REFERENCES sources(source_key),
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL
            );
        """)
        self.db.commit()
        self.db.execute("INSERT OR IGNORE INTO registry_meta(key,value) VALUES ('schema_version',?)",
                        (str(SCHEMA_VERSION),))
        saved = self.db.execute("SELECT value FROM registry_meta WHERE key='schema_version'").fetchone()[0]
        if saved != str(SCHEMA_VERSION):
            raise RuntimeError(f"Unsupported admission registry schema version: {saved}")
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _register_source(self, identity):
        key = source_key(identity)
        encoded = canonical_json(identity)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        existing = self.db.execute("SELECT identity_json FROM sources WHERE source_key=?", (key,)).fetchone()
        if existing and existing[0] != encoded:
            raise RuntimeError(f"Pinned source identity changed for {key}")
        self.db.execute("INSERT OR IGNORE INTO sources(source_key,identity_json,identity_sha256) VALUES (?,?,?)",
                        (key, encoded, digest))
        return key

    def _insert_document(self, source, record, mode, counters):
        doc_id = str(record["source_doc_id"])
        digest = str(record["normalized_hash"])
        if not doc_id or not digest:
            raise ValueError("Document source ID and normalized content hash are required")
        requested = record.get("split")
        split = "test" if mode == "sealed_test" else (
            requested if mode == "import" else content_split(digest))
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid document split: {split}")
        byte_count = int(record.get("utf8_bytes", 0))
        if byte_count < 0:
            raise ValueError("utf8_bytes cannot be negative")
        provenance = canonical_json(record.get("provenance", {}))

        prior_doc = self.db.execute(
            "SELECT normalized_hash,split FROM documents WHERE source_key=? AND source_doc_id=?",
            (source, doc_id)).fetchone()
        if prior_doc:
            if prior_doc["normalized_hash"] != digest or prior_doc["split"] != split:
                raise RuntimeError(f"Source document ID changed identity or split: {doc_id}")
            counters["already_registered"] += 1
            counters[f"{split}_documents"] += 1
            counters["utf8_bytes"] += byte_count
            return

        group = self.db.execute(
            "SELECT split,first_source_key,first_doc_id FROM content_groups WHERE normalized_hash=?",
            (digest,)).fetchone()
        if group:
            if group["split"] != split:
                raise RuntimeError(
                    f"Held-out overlap for normalized content {digest}: "
                    f"registered as {group['split']}, requested as {split}"
                )
            if mode != "import":
                reason = "sealed test overlap" if split == "test" else "exact duplicate"
                raise RuntimeError(f"{reason} for normalized content {digest}")
            sealed = self.db.execute(
                "SELECT value FROM registry_meta WHERE key='sealed_test_selection_id'").fetchone()
            if split == "test" and sealed:
                raise RuntimeError("The sealed test selection is immutable")
            self.db.execute("UPDATE content_groups SET document_count=document_count+1 WHERE normalized_hash=?",
                            (digest,))
        else:
            if split == "test":
                existing_seal = self.db.execute(
                    "SELECT value FROM registry_meta WHERE key='sealed_test_selection_id'").fetchone()
                if existing_seal and mode != "sealed_test":
                    raise RuntimeError("The sealed test selection is immutable")
            self.db.execute("INSERT INTO content_groups(normalized_hash,split,first_source_key,first_doc_id) VALUES (?,?,?,?)",
                            (digest, split, source, doc_id))

        self.db.execute("INSERT INTO documents(source_key,source_doc_id,normalized_hash,split,utf8_bytes,provenance_json) "
                        "VALUES (?,?,?,?,?,?)", (source, doc_id, digest, split, byte_count, provenance))
        counters["registered"] += 1
        counters[f"{split}_documents"] += 1
        counters["utf8_bytes"] += byte_count

    def ingest_documents(self, identity, records, mode="admit", dry_run=False,
                         sealed_test_selection_id=None):
        """Stream records into the registry in one transaction; rollback on any conflict."""
        if mode not in ("admit", "import", "sealed_test"):
            raise ValueError("mode must be admit, import, or sealed_test")
        if mode == "sealed_test" and not sealed_test_selection_id:
            raise ValueError("sealed_test requires an immutable selection ID")
        counters = {"registered": 0, "already_registered": 0, "train_documents": 0,
                    "validation_documents": 0, "test_documents": 0, "utf8_bytes": 0,
                    "records": 0}
        fingerprint = hashlib.sha256()
        seen_source_ids = set()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            source = self._register_source(identity)
            if mode == "sealed_test":
                existing = self.db.execute(
                    "SELECT value FROM registry_meta WHERE key='sealed_test_selection_id'").fetchone()
                if existing and existing[0] != sealed_test_selection_id:
                    raise RuntimeError("A different sealed test selection is already registered")
            for record in records:
                counters["records"] += 1
                digest = str(record["normalized_hash"] if "normalized_hash" in record
                             else content_hash(record["text"]))
                normalized = {**record, "normalized_hash": digest}
                source_doc_id = str(record["source_doc_id"])
                source_id_pair = (source, source_doc_id)
                if mode != "import" and source_id_pair in seen_source_ids:
                    raise RuntimeError(f"Duplicate source document ID in candidate stream: {source_doc_id}")
                seen_source_ids.add(source_id_pair)
                self._insert_document(source, normalized, mode, counters)
                fingerprint.update(canonical_json([str(record["source_doc_id"]), digest,
                    ("test" if mode == "sealed_test" else record.get("split"))]).encode("utf-8") + b"\n")
            fingerprint_value = fingerprint.hexdigest()
            if mode == "sealed_test":
                prior_fingerprint = self.db.execute(
                    "SELECT value FROM registry_meta WHERE key='sealed_test_fingerprint'").fetchone()
                if prior_fingerprint and prior_fingerprint[0] != fingerprint_value:
                    raise RuntimeError("Sealed test content differs from its original registration")
                self.db.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES ('sealed_test_selection_id',?)",
                                (sealed_test_selection_id,))
                self.db.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES ('sealed_test_source_key',?)",
                                (source,))
                self.db.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES ('sealed_test_fingerprint',?)",
                                (fingerprint_value,))
            counters["content_fingerprint"] = fingerprint_value
            if dry_run:
                self.db.rollback()
            else:
                self.db.commit()
            return counters
        except BaseException:
            self.db.rollback()
            raise

    def register_selection(self, selection):
        """Persist a canonical immutable selection manifest (idempotent by selection ID)."""
        selection_id = selection["selection_id"]
        key = selection["source_key"]
        encoded = canonical_json(selection)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            identity = selection["source_identity"]
            source = self._register_source(identity)
            if source != key:
                raise RuntimeError("Selection source key does not match its pinned identity")
            prior = self.db.execute("SELECT source_key,manifest_json FROM selections WHERE selection_id=?",
                                    (selection_id,)).fetchone()
            if prior:
                old = json.loads(prior["manifest_json"])
                stable_keys = ("source_key", "source_identity", "manifest_sha256", "byte_budget",
                               "selected_files", "selection")
                if prior["source_key"] != source or any(old.get(name) != selection.get(name) for name in stable_keys):
                    raise RuntimeError("Selection ID is already bound to a different immutable selection")
                old_counters = old.get("document_counters", {})
                new_counters = selection.get("document_counters", {})
                deterministic_counters = ("records", "train_documents", "validation_documents",
                                          "test_documents", "utf8_bytes", "content_fingerprint")
                if any(old_counters.get(name) != new_counters.get(name) for name in deterministic_counters):
                    raise RuntimeError("Selection ID is already bound to different document counters")
                self.db.commit()
                return old
            self.db.execute("INSERT INTO selections(selection_id,source_key,manifest_json,manifest_sha256) VALUES (?,?,?,?)",
                            (selection_id, source, encoded, digest))
            self.db.commit()
            return selection
        except BaseException:
            self.db.rollback()
            raise

    def import_sidecars(self, token_root, identity=None):
        """Import bounded JSONL provenance from existing prepared chunks/shards."""
        root = Path(token_root)
        totals = {"sidecars": 0, "records": 0, "registered": 0, "already_registered": 0,
                  "train_documents": 0, "validation_documents": 0, "test_documents": 0,
                  "utf8_bytes": 0}
        manifests = sorted(p for p in root.rglob("manifest.json")
                           if not p.parent.name.endswith(".partial") and p.parent != root)
        for manifest_path in manifests:
            meta = json.loads(manifest_path.read_text())
            revision = meta.get("source_revision")
            if identity is not None:
                if identity.get("revision") != revision:
                    raise RuntimeError(f"Sidecar revision {revision} does not match pinned identity")
                doc_identity = identity
            else:
                doc_identity = {"namespace": "legacy-sidecar-import", "revision": revision,
                                "tokenizer_sha256": meta.get("tokenizer_sha256")}
            for split in ("train", "validation", "test"):
                sidecar = manifest_path.parent / f"{split}.documents.jsonl"
                if not sidecar.exists():
                    continue

                def records(path=sidecar, split_name=split, meta_info=meta):
                    with path.open() as handle:
                        for line_number, line in enumerate(handle, 1):
                            row = json.loads(line)
                            if "id" not in row or "content_hash" not in row:
                                raise ValueError(f"Missing source ID/content hash in {path}:{line_number}")
                            yield {"source_doc_id": stable_source_doc_id(row["id"]),
                                   "normalized_hash": row["content_hash"], "split": split_name,
                                   "utf8_bytes": int(row.get("utf8_bytes", 0)),
                                   "provenance": {"sidecar": str(path), "source": meta_info.get("source"),
                                                  "offset": row.get("offset"), "tokens": row.get("tokens")}}

                counts = self.ingest_documents(doc_identity, records(), mode="import")
                totals["sidecars"] += 1
                for key in ("records", "registered", "already_registered", "train_documents",
                            "validation_documents", "test_documents", "utf8_bytes"):
                    totals[key] += counts[key]
        return totals

    def register_selection_file(self, selection, output=None):
        stored = self.register_selection(selection)
        output = Path(output) if output else self.path.parent / "selections" / f"{selection['selection_id']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(stored, indent=2, ensure_ascii=False) + "\n"
        with (output.parent / ".selection.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.exists():
                if output.read_text() != serialized:
                    raise RuntimeError(f"Selection manifest is immutable and already differs: {output}")
            else:
                temp = output.with_suffix(output.suffix + ".partial")
                temp.write_text(serialized)
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                temp.replace(output)
        return output


def _local_records(data_dir, selected):
    base = Path(data_dir)
    for item in selected:
        path = base / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Selected source file missing or wrong size: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Selected source file checksum mismatch: {path}")
        for doc in documents(path):
            if doc.get("id") is None or not isinstance(doc.get("text"), str):
                raise RuntimeError(f"Document lacks a stable ID or text: {path}")
            doc_id = stable_source_doc_id(doc["id"])
            text = doc["text"]
            yield {"source_doc_id": doc_id, "text": text,
                   "utf8_bytes": len(text.encode("utf-8")),
                   "provenance": {"file": item["path"]}}


def _copy_registry_for_dry_run(path, directory):
    """Create a consistent disposable SQLite copy, including any active WAL state."""
    destination = Path(directory) / "registry.sqlite3"
    if Path(path).exists():
        source = sqlite3.connect(str(path), timeout=60)
        target = sqlite3.connect(str(destination), timeout=60)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    return destination


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", type=Path, default=data_root() / "admission/registry.sqlite3")
    common.add_argument("--manifest", type=Path, required=True, help="Pinned raw-data manifest JSON")
    common.add_argument("--byte-budget", type=int, required=True, help="Maximum selected raw bytes")
    common.add_argument("--reserve-bytes", type=int, required=True, help="Free bytes to retain after selection")
    common.add_argument("--filesystem", type=Path, required=True, help="Target filesystem checked for free space")
    dry = sub.add_parser("dry-run", parents=[common], help="Plan a deterministic, byte-bounded selection")
    dry.add_argument("--data-dir", type=Path, help="Optionally validate and stream local files without registry writes")
    admit = sub.add_parser("admit", parents=[common], help="Stream local files into the identity registry")
    admit.add_argument("--data-dir", type=Path, required=True, help="Existing local snapshot root; no download is performed")
    admit.add_argument("--selection-output", type=Path)
    test = sub.add_parser("seal-test", parents=[common], help="Register a fixed held-out test and seal it")
    test.add_argument("--data-dir", type=Path, required=True)
    test.add_argument("--selection-output", type=Path)
    imports = sub.add_parser("import-sidecars", help="Import prior prepared document provenance")
    imports.add_argument("--registry", type=Path, default=data_root() / "admission/registry.sqlite3")
    imports.add_argument("--tokens-root", type=Path, required=True)
    imports.add_argument("--identity-manifest", type=Path,
                         help="Raw manifest for a single pinned dataset identity")
    imports.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "import-sidecars":
        identity = None
        if args.identity_manifest:
            manifest = json.loads(args.identity_manifest.read_text())
            identity = source_identity(manifest)
        if args.dry_run:
            with tempfile.TemporaryDirectory(prefix="microllama-admission-") as directory:
                copy = _copy_registry_for_dry_run(args.registry, directory)
                with AdmissionRegistry(copy) as registry:
                    result = registry.import_sidecars(args.tokens_root, identity=identity)
        else:
            with AdmissionRegistry(args.registry) as registry:
                result = registry.import_sidecars(args.tokens_root, identity=identity)
        print(json.dumps({"dry_run": args.dry_run, **result}, indent=2), flush=True)
        return

    manifest = json.loads(args.manifest.read_text())
    filesystem = args.filesystem.resolve()
    free_bytes = shutil.disk_usage(filesystem).free
    selection = build_selection(manifest, args.byte_budget, free_bytes,
                                args.reserve_bytes, filesystem)
    if args.command == "dry-run":
        local_result = None
        if args.data_dir:
            with tempfile.TemporaryDirectory(prefix="microllama-admission-") as directory:
                copy = _copy_registry_for_dry_run(args.registry, directory)
                with AdmissionRegistry(copy) as registry:
                    local_result = registry.ingest_documents(
                        selection["source_identity"],
                        _local_records(args.data_dir, selection["selected_files"]),
                        mode="admit")
        print(json.dumps({**selection, "local_scan": local_result}, indent=2), flush=True)
        return

    identity = selection["source_identity"]
    mode = "sealed_test" if args.command == "seal-test" else "admit"
    with AdmissionRegistry(args.registry) as registry:
        counts = registry.ingest_documents(identity, _local_records(args.data_dir, selection["selected_files"]),
            mode=mode, sealed_test_selection_id=selection["selection_id"] if mode == "sealed_test" else None)
        selection["document_counters"] = counts
        output = registry.register_selection_file(selection, args.selection_output)
    print(json.dumps({"selection_id": selection["selection_id"], "selection_manifest": str(output),
                      "mode": mode, "document_counters": counts,
                      "warning": "Selection is registered but is not yet integrated with token preparation."},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
