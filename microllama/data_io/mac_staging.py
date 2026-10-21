"""Pinned, lazy data restoration for an isolated Mac continuation.

The historical recipe and reader state stay unchanged. Only source locations
and the declared finite target change in a separately recorded catalog.
No training, source acquisition, or instance lifecycle action is performed here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

DATA_REVISION = "af9e5b1fa4455991d46aa2c4a623b905942b8053"
PUBLIC_REPO = "keeeeenw/MicroLlama2"
CHUNK_FILES = frozenset({"manifest.json", "train.bin", "train.documents.jsonl",
                         "validation.bin", "validation.documents.jsonl"})


def payload_files(directory, expected):
    """Ignore only bounded, recognizable AppleDouble metadata on ExFAT.

    Archive members remain strictly allowlisted. macOS can create ._ sidecars
    after ordinary local writes; they are never training payload or provenance.
    """
    expected = set(expected)
    result = {}
    for path in Path(directory).iterdir():
        if path.name in expected:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Payload hash verification requires a regular file')
            result[path.name] = path
        elif path.name.startswith('._') and path.name[2:] in expected:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
                raise ValueError('Invalid local AppleDouble sidecar')
            with path.open('rb') as stream:
                header = stream.read(8)
            if header not in (bytes.fromhex('0005160700020000'), bytes.fromhex('0005160700010000')):
                raise ValueError('Unrecognized local metadata sidecar')
        else:
            raise ValueError('Unexpected payload directory member')
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _component(value):
    if (not isinstance(value, str) or not value or value in (".", "..") or
            "/" in value or "\\" in value or value.endswith(".partial")):
        raise ValueError("Unsafe source or chunk name")
    return value


def _write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_reader_state(catalog, state):
    """Check cursor identities without changing them or opening token payloads."""
    recipe = catalog["recipe"]
    for key, value in (("mode", catalog["mode"]), ("recipe_id", catalog["recipe_id"]),
                       ("sequence_length", recipe["sequence_length"]), ("seed", recipe["seed"])):
        if state.get(key) != value:
            raise ValueError(f"Reader identity differs: {key}")
    names = set(catalog["sources"])
    if set(state["sources"]) != names or set(state["source_targets"]) != names:
        raise ValueError("Reader source set differs")
    for source, entry in catalog["sources"].items():
        cursor = state["sources"][source]
        expected = {k: v["manifest_sha256"] for k, v in entry["chunks"].items()}
        if cursor["known"] != expected:
            raise ValueError("Reader inventory differs from frozen complete catalog")
        consumed = cursor["consumed"]
        if len(set(consumed)) != len(consumed) or not set(consumed).issubset(expected):
            raise ValueError("Invalid consumed chunk list")
        current = cursor["current"]
        if current is not None and (current not in expected or current in consumed):
            raise ValueError("Invalid current chunk")
        if (type(cursor["cursor"]) is not int or cursor["cursor"] < 0 or
                (current is None and cursor["cursor"] != 0)):
            raise ValueError("Invalid source cursor")
        total = sum(v["valid_targets"] for v in entry["chunks"].values())
        count = state["source_targets"][source]
        if type(count) is not int or not 0 <= count <= total:
            raise ValueError("Invalid consumed target counter")
    parent = state.get("parent_tokens")
    if type(parent) is not int or parent < 0:
        raise ValueError("Invalid historical token counter")
    origin = state.get("policy_origin")
    if origin is not None:
        if (set(origin) != names or any(type(v) is not int or not 0 <= v <= state["source_targets"][s]
                                        for s, v in origin.items())):
            raise ValueError("Invalid mixture origin")
        transition = state.get("mixture_transition", {})
        if (not isinstance(transition, dict) or any(
                not isinstance(transition.get(k), str) or len(transition[k]) != 64 or
                any(c not in "0123456789abcdef" for c in transition[k])
                for k in ("parent_recipe_id", "parent_catalog_sha256"))):
            raise ValueError("Invalid mixture transition binding")
        if transition.get("starting_tokens") != parent + sum(origin.values()):
            raise ValueError("Mixture origin does not reconcile")
    elif "mixture_transition" in state:
        raise ValueError("Mixture transition requires a saved policy origin")
    return parent + sum(state["source_targets"].values())


def derive_catalog(parent_catalog, destination, source_root, *, target_tokens,
                   expected_parent_sha256, reader_state=None):
    """Write a new catalog with only source paths and finite target changed.

    Absolute legacy paths inside the recipe deliberately remain historical
    identity strings. Loading an existing mixture reader never uses those paths.
    Returns the immutable derivation receipt; an existing destination is refused.
    """
    from .growth_reader import load_catalog

    parent_catalog, destination = Path(parent_catalog), Path(destination)
    parent_pointer = _read(parent_catalog / "manifest.json")
    if parent_pointer["sha256"] != expected_parent_sha256:
        raise ValueError("Parent catalog SHA differs")
    original = load_catalog(parent_catalog, expected_sha256=expected_parent_sha256)
    if not original.get("complete") or original.get("status") != "complete":
        raise ValueError("A frozen complete parent catalog is required")
    if type(target_tokens) is not int or target_tokens <= original["target_tokens"]:
        raise ValueError("A new explicit target above the historical catalog budget is required")
    source_root = Path(source_root).absolute()
    if destination.resolve() == parent_catalog.resolve() or parent_catalog.resolve() in destination.resolve().parents:
        raise ValueError("Derived catalog must be separate from its parent")
    result = copy.deepcopy(original)
    locations = {}
    for source, entry in result["sources"].items():
        _component(source)
        old = Path(entry["directory"]).resolve()
        new = (source_root / source / "tokens").resolve()
        if new == old or old in new.parents:
            raise ValueError("Mac data staging must not modify historical source locations")
        locations[source] = {"before": entry["directory"], "after": str(new)}
        entry["directory"] = str(new)
    result["target_tokens"] = target_tokens
    # Refuse accidental recipe/pin/inventory rewrites before creating any output.
    restored = copy.deepcopy(result)
    restored["target_tokens"] = original["target_tokens"]
    for source, paths in locations.items():
        restored["sources"][source]["directory"] = paths["before"]
    if restored != original:
        raise ValueError("Unapproved derived-catalog difference")
    receipt = {"schema_version": 1, "purpose": "isolated-mac-continuation",
               "training_started": False, "target_authorization": "not established by data staging",
               "parent_catalog_sha256": expected_parent_sha256,
               "parent_catalog": str(parent_catalog.resolve()),
               "recipe_id": result["recipe_id"], "source_locations": locations,
               "target_tokens_before": original["target_tokens"], "target_tokens_after": target_tokens}
    if reader_state is not None:
        current = validate_reader_state(original, reader_state)
        origin = reader_state.get("policy_origin", {s: 0 for s in original["sources"]})
        counts = {s: reader_state["source_targets"][s] - origin[s] for s in origin}
        elapsed = sum(counts.values())
        capacity = min((sum(v["valid_targets"] for v in original["sources"][s]["chunks"].values())
                        - origin[s]) / original["recipe"]["sources"][s]["weight"]
                       for s in origin) - elapsed
        if target_tokens <= current or target_tokens - current > capacity - recipe_margin(original):
            raise ValueError("New target exceeds balanced unconsumed data or lacks one-update margin")
        receipt.update(reader_state_sha256=_canonical_sha(reader_state), parent_tokens=current,
                       balanced_unconsumed_targets=int(capacity))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "snapshots").mkdir()
    snapshot = destination / "snapshots" / ("mac-" + _canonical_sha(result)[:16] + ".json")
    _write_new(snapshot, result)
    checksum = _sha(snapshot)
    _write_new(destination / "manifest.json", {"mode": original["mode"], "snapshot": snapshot.name,
               "sha256": checksum, "generation": original["generation"]})
    receipt["derived_catalog_sha256"] = checksum
    _write_new(destination / "derivation.json", receipt)
    return receipt


def recipe_margin(catalog):
    # Historical global batch:16 rows, regardless of the number of GPU ranks.
    return catalog["recipe"]["sequence_length"] * 16


class ArchiveStager:
    """Stage exact published chunks lazily; never advance or reset a reader."""
    def __init__(self, catalog_dir, archives_path, files_path, cache_dir, *,
                 data_revision=DATA_REVISION, fetch_archive=None, reserve_bytes=10 * 2**30,
                 max_download_bytes=0, existing_archive_roots=(), extra_source_roots=None):
        from .growth_reader import load_catalog

        if data_revision != DATA_REVISION:
            raise ValueError("Only the pinned published data revision is supported")
        self.catalog_dir = Path(catalog_dir)
        self.catalog = load_catalog(catalog_dir)
        self.inventory = _read(files_path)
        index = _read(archives_path)
        if index.get("all_remote_sha256_verified") is not True:
            raise ValueError("Unverified archive index")
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("Invalid disk reserve")
        if type(max_download_bytes) is not int or max_download_bytes < 0:
            raise ValueError("Invalid download budget")
        self.reserve_bytes = reserve_bytes
        self.cache_dir = Path(cache_dir)
        self.max_download_bytes = max_download_bytes
        self.download_bytes = 0
        self.existing_archive_roots = [Path(p) for p in existing_archive_roots] + [self.cache_dir]
        self.data_revision = data_revision
        self.fetch_archive = fetch_archive
        self.entries = copy.deepcopy(self.catalog['sources'])
        for source, directory in (extra_source_roots or {}).items():
            _component(source)
            if source in self.entries or source not in self.inventory:
                raise ValueError('Invalid extra frozen validation source')
            self.entries[source] = {'directory': str(Path(directory).absolute()),
                'chunks': {name: {'manifest_sha256': files['manifest.json']['sha256']}
                           for name, files in self.inventory[source].items()}}
        self.archives = {}
        for item in index["archives"]:
            path = PurePosixPath(item["path_in_repo"])
            if len(path.parts) != 4 or path.parts[:2] != ("datasets", "tokenized-v1"):
                raise ValueError("Unsafe or duplicate archive entry")
            key = str(path)
            if key in self.archives:
                raise ValueError("Duplicate archive entry")
            self.archives[key] = item
        self.parts = {}
        for source, entry in self.entries.items():
            _component(source)
            if source not in self.inventory or set(entry["chunks"]) != set(self.inventory[source]):
                raise ValueError("Published chunk inventory differs from catalog")
            names = sorted(entry["chunks"])
            for position, name in enumerate(names):
                _component(name)
                files = self.inventory[source][name]
                if set(files) != CHUNK_FILES or files["manifest.json"]["sha256"] != entry["chunks"][name]["manifest_sha256"]:
                    raise ValueError("Published chunk manifest differs from catalog")
                part = position // 50
                relative = f"datasets/tokenized-v1/{source}/part-{part:05d}.tar.zst"
                if relative not in self.archives:
                    raise ValueError("Missing published archive")
                group = names[part * 50:(part + 1) * 50]
                if self.archives[relative]["chunks"] != len(group):
                    raise ValueError("Published archive chunk count differs")
                self.parts[source, name] = relative, group

    def _verify_chunk(self, source, name):
        folder = Path(self.entries[source]["directory"]) / name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Chunk destination is not a regular directory")
        if set(payload_files(folder, CHUNK_FILES)) != CHUNK_FILES:
            raise ValueError("Restored chunk file set differs")
        for filename, expected in self.inventory[source][name].items():
            path = folder / filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
                raise ValueError("Restored chunk hash differs")
        return folder

    def _local_archive(self, item):
        """Look at explicit locations and Hub's local index; never use HTTP."""
        if self.fetch_archive is not None:
            # Injectable LOCAL resolver for tests or a caller-owned archive map.
            # It must return a local file, never perform acquisition itself.
            path = self.fetch_archive(item)
            if path is not None:
                return Path(path)
        relative = Path(item["path_in_repo"])
        for root in self.existing_archive_roots:
            candidates = (root / relative, root / relative.parts[-2] / relative.name,
                          root / "models--keeeeenw--MicroLlama2" / "blobs" / item["sha256"],
                          root / "hf" / "models--keeeeenw--MicroLlama2" / "blobs" / item["sha256"])
            for path in candidates:
                if path.is_file() and path.stat().st_size == item["bytes"]:
                    return path
        try:
            from huggingface_hub import try_to_load_from_cache
            for cache in (self.cache_dir / "hf", None):
                path = try_to_load_from_cache(PUBLIC_REPO, item["path_in_repo"],
                                             revision=self.data_revision,
                                             cache_dir=str(cache) if cache is not None else None)
                if (isinstance(path, str) and Path(path).is_file() and
                        Path(path).stat().st_size == item["bytes"]):
                    return Path(path)
        except ImportError:
            pass
        return None

    def _download(self, item):
        path = self._local_archive(item)
        if path is None:
            if item["bytes"] > self.max_download_bytes - self.download_bytes:
                raise ValueError(f"Archive needs {item['bytes']} download bytes; explicit remaining budget is "
                                 f"{self.max_download_bytes - self.download_bytes}")
            # Count conservatively before the request. A failed/interrupted
            # attempt is not an implicit license to retry within this session.
            self.download_bytes += item["bytes"]
            os.environ['HF_XET_CACHE'] = str(self.cache_dir / 'xet')
            os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(PUBLIC_REPO, item["path_in_repo"], revision=self.data_revision,
                                   token=False, cache_dir=str(self.cache_dir / "hf"))
        # HF snapshots may be symlinks to immutable cache blobs. Resolve the input
        # archive only; extracted members and output chunks must never be links.
        path = Path(path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise ValueError("Compressed archive hash differs")
        return path

    def ensure_chunk(self, source, name):
        _component(source); _component(name)
        if (source, name) not in self.parts:
            raise ValueError("Chunk is outside the pinned inventory")
        root = Path(self.entries[source]["directory"])
        folder = root / name
        if folder.exists() or folder.is_symlink():
            return self._verify_chunk(source, name)
        relative, group = self.parts[source, name]
        item = self.archives[relative]
        root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        expected = {f"{chunk}/{filename}": info for chunk in group
                    for filename, info in self.inventory[source][chunk].items()}
        payload_bytes = sum(x["bytes"] for x in expected.values())
        if (shutil.disk_usage(root).free < self.reserve_bytes + payload_bytes or
                shutil.disk_usage(self.cache_dir).free < self.reserve_bytes + payload_bytes + item["bytes"]):
            raise ValueError("Insufficient free disk space for verified atomic staging")
        archive = self._download(item)
        with tempfile.TemporaryDirectory(prefix=".mac-stage-", dir=root) as temporary:
            temporary = Path(temporary)
            decoder = subprocess.Popen(["zstd", "-d", "-c", str(archive)], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            seen = set()
            try:
                with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
                    for member in tar:
                        path = PurePosixPath(member.name)
                        if (path.is_absolute() or ".." in path.parts or "\\" in member.name or
                                str(path) != member.name or not path.parts):
                            raise ValueError("Unsafe archive member path")
                        if member.isdir():
                            if len(path.parts) != 1 or path.name not in group:
                                raise ValueError("Unexpected archive directory")
                            continue
                        if not member.isfile() or member.name in seen:
                            raise ValueError("Archive links/devices/duplicates are forbidden")
                        seen.add(member.name)
                        # The publisher optionally included a source summary in
                        # part zero. It is bound by the compressed archive hash,
                        # but is never installed or used as chunk provenance.
                        if member.name == "manifest.json" and relative.endswith("part-00000.tar.zst"):
                            if member.size > 16 * 1024**2:
                                raise ValueError("Oversized source summary")
                            continue
                        if member.name not in expected or member.size != expected[member.name]["bytes"]:
                            raise ValueError("Archive member inventory differs")
                        output = temporary.joinpath(*path.parts)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        digest = hashlib.sha256()
                        with tar.extractfile(member) as stream, output.open("xb") as writer:
                            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                                writer.write(block); digest.update(block)
                        if output.stat().st_size != member.size or digest.hexdigest() != expected[member.name]["sha256"]:
                            raise ValueError("Archive payload hash differs")
                if not set(expected).issubset(seen):
                    raise ValueError("Archive is missing required chunk members")
                _, error = decoder.communicate()
                if decoder.returncode:
                    raise ValueError("Archive decompression failed: " + error.decode(errors="replace")[:200])
            finally:
                if decoder.poll() is None:
                    decoder.kill(); decoder.communicate()
            # All archive payload hashes passed before making any chunk visible.
            # Each directory rename is atomic on the source filesystem. An
            # interrupted group leaves only independently verified whole chunks.
            for chunk in group:
                target = root / chunk
                if target.exists() or target.is_symlink():
                    self._verify_chunk(source, chunk)
            for chunk in group:
                target = root / chunk
                if not target.exists():
                    os.rename(temporary / chunk, target)
            return self._verify_chunk(source, name)

    def estimate_current(self, reader_state):
        """Return transfer estimates without downloads or broad filesystem scans.

        Local archive candidates are checked by size here; ensure_chunk performs
        the mandatory full archive and payload hashes before installing them.
        """
        validate_reader_state(self.catalog, reader_state)
        current, missing = {}, {}
        for source, cursor in reader_state["sources"].items():
            name = cursor["current"]
            if name is None:
                candidates = sorted(set(cursor["known"]) - set(cursor["consumed"]))
                if not candidates:
                    raise ValueError(f"No unconsumed chunk remains for {source}")
                name = candidates[0]
            current[source] = name
            folder = Path(self.catalog["sources"][source]["directory"]) / name
            if folder.is_dir() and not folder.is_symlink():
                continue
            relative, _ = self.parts[source, name]
            item = self.archives[relative]
            local = self._local_archive(item)
            missing[relative] = {"bytes": item["bytes"],
                                 "local_candidate": str(local) if local is not None else None,
                                 "download_bytes": 0 if local is not None else item["bytes"]}
        return {"current_chunks": current, "archives": missing,
                "download_bytes_needed": sum(x["download_bytes"] for x in missing.values()),
                "max_download_bytes": self.max_download_bytes,
                "download_bytes_already_requested": self.download_bytes}

    def estimate_chunks(self, chunks):
        """Unique archive cost for explicit (source,chunk) pairs, without HTTP."""
        missing = {}
        for source, name in chunks:
            if (source, name) not in self.parts:
                raise ValueError('Requested chunk is outside pinned inventory')
            folder = Path(self.entries[source]['directory']) / name
            if folder.is_dir() and not folder.is_symlink():
                continue
            relative, _ = self.parts[source, name]
            item = self.archives[relative]
            local = self._local_archive(item)
            missing[relative] = {'bytes': item['bytes'],
                'local_candidate': str(local) if local is not None else None,
                'download_bytes': 0 if local is not None else item['bytes']}
        return {'archives': missing, 'download_bytes_needed': sum(
            value['download_bytes'] for value in missing.values()),
            'max_download_bytes': self.max_download_bytes}

    def stage_chunks(self, chunks):
        chunks = list(chunks)
        estimate = self.estimate_chunks(chunks)
        if estimate['download_bytes_needed'] > self.max_download_bytes - self.download_bytes:
            raise ValueError('Frozen chunks exceed the explicit remaining download budget')
        return {f'{source}/{name}': str(self.ensure_chunk(source, name))
                for source, name in chunks}

    def stage_current(self, reader_state):
        estimate = self.estimate_current(reader_state)
        if estimate["download_bytes_needed"] > self.max_download_bytes - self.download_bytes:
            raise ValueError(f"Current chunks need {estimate['download_bytes_needed']} download bytes; "
                             "increase the explicit budget or supply verified local archives")
        staged = {}
        for source, name in estimate["current_chunks"].items():
            staged[source] = str(self.ensure_chunk(source, name))
        return staged

    def reader(self, reader_state):
        """Return the original reader with staging at its file-open boundary."""
        from .growth_reader import MixtureTokenBlocks
        stager = self

        class StagedMixtureTokenBlocks(MixtureTokenBlocks):
            def _open(self, source, name):
                stager.ensure_chunk(source, name)
                return super()._open(source, name)

        validate_reader_state(self.catalog, reader_state)
        reader = StagedMixtureTokenBlocks(self.catalog_dir, self.catalog["recipe"]["sequence_length"],
                                          seed=self.catalog["recipe"]["seed"])
        reader.load_state_dict(copy.deepcopy(reader_state))
        if reader.state_dict() != reader_state:
            raise ValueError("Restored reader state differs")
        return reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalog", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Verified training_state.pt")
    parser.add_argument("--checkpoint-sha256", required=True, help="SHA from the verified checkpoint manifest")
    parser.add_argument("--output", type=Path, required=True, help="New isolated derived-catalog directory")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--stage-current", action="store_true")
    parser.add_argument("--max-download-bytes", type=int, default=0,
                        help="Per-invocation archive download ceiling; default0 permits local files only")
    parser.add_argument("--existing-archive-root", type=Path, action="append", default=[],
                        help="Existing archive/Hub-cache root; checked before any transfer")
    args = parser.parse_args()
    if _sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint training-state hash differs")
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["data"]
    if checkpoint["tokens"] != state["parent_tokens"] + sum(state["source_targets"].values()):
        raise ValueError("Checkpoint tokens do not reconcile with source state")
    receipt = derive_catalog(args.parent_catalog, args.output, args.source_root,
                             target_tokens=args.target_tokens, expected_parent_sha256=args.parent_sha256,
                             reader_state=state)
    stager = ArchiveStager(args.output, args.archives, args.files, args.cache_dir,
                          max_download_bytes=args.max_download_bytes,
                          existing_archive_roots=args.existing_archive_root)
    receipt["transfer_plan"] = stager.estimate_current(state)
    _write_new(args.output / "transfer-plan.json", receipt["transfer_plan"])
    if args.stage_current:
        receipt["staged_current"] = stager.stage_current(state)
        _write_new(args.output / "staged-current.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
