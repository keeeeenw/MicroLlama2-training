"""Download the pinned FineWeb-Edu sample to the standard Hugging Face cache."""
import argparse
import hashlib
import json
from pathlib import Path

from ..common.paths import data_root
from .transfer import atomic_json, file_identity

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PREFIX = "sample/10BT/"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Resolve the bounded sample manifest without downloading")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, help="Override the standard HF cache; HF_HOME is also respected")
    parser.add_argument("--output", type=Path, help="Optionally copy the cached sample to this external folder")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    from huggingface_hub import HfApi, snapshot_download
    manifest_path = data_root() / "raw/fineweb-edu-10bt/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        files = [{"path": file.rfilename, "bytes": file.size, "sha256": file.lfs.sha256 if file.lfs else None}
                 for file in info.siblings if file.rfilename.startswith(PREFIX)
                 and file.rfilename.endswith(".parquet")]
        if not files or any(not file["bytes"] or not file["sha256"] for file in files):
            raise RuntimeError("Could not resolve the official sample checksums")
        manifest = {"repo": REPO, "revision": info.sha, "config": "sample-10BT", "files": files,
                    "total_bytes": sum(file["bytes"] for file in files)}
        atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
        return
    from huggingface_hub.constants import HF_HUB_CACHE
    # HF publishes each finished shard atomically here. Preparation may consume it
    # immediately after its own size/SHA verification, without waiting for other shards.
    cache = args.cache_dir or Path(HF_HUB_CACHE)
    expected = cache / ("datasets--" + manifest["repo"].replace("/", "--")) / "snapshots" / manifest["revision"]
    manifest.update(data_dir=str(expected.resolve()), download_complete=False,
                    transfer_complete=args.output is None, verified=False)
    atomic_json(manifest_path, manifest)
    print(f"Downloading {len(manifest['files'])} official shards to the Hugging Face cache.", flush=True)
    snapshot = Path(snapshot_download(repo_id=manifest["repo"], repo_type="dataset", revision=manifest["revision"],
        allow_patterns=[file["path"] for file in manifest["files"]], cache_dir=args.cache_dir, max_workers=args.workers))
    # Record the snapshot pointer; never modify, rename or remove HF cache files.
    manifest.update(data_dir=str(snapshot.resolve()), download_complete=True)
    atomic_json(manifest_path, manifest)
    for item in manifest["files"]:
        path = snapshot / item["path"]
        identity = file_identity(path)
        if identity["bytes"] != item["bytes"]:
            raise RuntimeError(f"Downloaded shard size mismatch: {path}")
        if (item.get("cache_identity") != identity or item.get("cache_verified_sha256") != item["sha256"]) and sha256(path) != item["sha256"]:
            raise RuntimeError(f"Downloaded shard checksum mismatch: {path}")
        item.update(cache_identity=identity, cache_verified_sha256=item["sha256"])
        print(f"Verified {item['path']}", flush=True)
    manifest.update(download_complete=True, verified=True)
    atomic_json(manifest_path, manifest)
    print(f"Download complete. Dataset available at {snapshot}", flush=True)
    if args.output:
        from .transfer import transfer
        transfer(manifest, snapshot, args.output, manifest_path)


if __name__ == "__main__":
    main()
