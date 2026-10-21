"""Local archive and cursor tests; no Hub calls, production data, or training."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from microllama.data_io.growth_reader import MixtureTokenBlocks, digest, load_catalog, publish_catalog, usable_targets
from microllama.data_io.mac_staging import ArchiveStager, derive_catalog, CHUNK_FILES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def compress_archive(path, members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, payload in members:
            item = tarfile.TarInfo(name)
            if payload is None:
                item.type = tarfile.SYMTYPE
                item.linkname = "../../escape"
                tar.addfile(item)
            else:
                item.size = len(payload)
                tar.addfile(item, io.BytesIO(payload))
    result = subprocess.run(["zstd", "-q", "-c"], input=stream.getvalue(), capture_output=True, check=True)
    path.write_bytes(result.stdout)


@pytest.fixture
def corpus(tmp_path):
    names = ("edu", "dclm", "cosmopedia", "finemath")
    sources, inventory, archives, members, paths = {}, {}, [], {}, {}
    for number, source in enumerate(names):
        root = tmp_path / "original" / source
        source_chunks, inventory[source], members[source] = {}, {}, []
        for chunk in ("a", "b"):
            folder = root / chunk
            folder.mkdir(parents=True)
            meta = {"source_revision": source, "tokenizer_sha256": "tok", "splits": {}}
            for split in ("train", "validation"):
                values = np.arange(1000 + number * 1000, 1000 + number * 1000 + 81, dtype="<u2")
                (folder / f"{split}.bin").write_bytes(values.tobytes())
                (folder / f"{split}.documents.jsonl").write_text('{"id":"fixture"}\n')
                meta["splits"][split] = {"tokens": len(values), "sha256": sha(folder / f"{split}.bin")}
            write_json(folder / "manifest.json", meta)
            inventory[source][chunk] = {file.name: {"bytes": file.stat().st_size, "sha256": sha(file)}
                                        for file in folder.iterdir() if file.name in CHUNK_FILES}
            source_chunks[chunk] = {"manifest_sha256": sha(folder / "manifest.json"),
                                    "valid_targets": usable_targets(folder / "train.bin", 4)}
            members[source].extend((f"{chunk}/{file.name}", file.read_bytes()) for file in sorted(folder.iterdir())
                                   if file.name in CHUNK_FILES)
        sources[source] = {"directory": str(root), "chunks": source_chunks}
        archive = tmp_path / f"{source}.tar.zst"
        compress_archive(archive, members[source])
        relative = f"datasets/tokenized-v1/{source}/part-00000.tar.zst"
        archives.append({"path_in_repo": relative, "chunks": 2, "bytes": archive.stat().st_size,
                         "sha256": sha(archive), "revision": "historical-upload-commit"})
        paths[relative] = archive
    recipe = {"sources": {source: {"weight": .25, "revision": source} for source in names},
              "sequence_length": 4, "seed": 42, "eos": 7, "pad": 0, "tokenizer_sha256": "tok",
              "legacy": {"root": "/workspace/original/tokens", "inventory": "/workspace/legacy.json",
                         "inventory_sha256": "a" * 64}}
    original = tmp_path / "catalog"
    publish_catalog(original, {"schema_version": 1, "mode": "mixture-growth-v1", "recipe": recipe,
                    "recipe_id": digest(recipe), "initial_tokens": 100, "target_tokens": 200,
                    "status": "complete", "complete": True, "sources": sources})
    reader = MixtureTokenBlocks(original, 4)
    state = reader.state_dict()
    state.update(parent_tokens=100, policy_origin={s: 0 for s in names},
                 mixture_transition={"parent_recipe_id": "a" * 64, "parent_catalog_sha256": "b" * 64,
                                     "starting_tokens": 100})
    reader.load_state_dict(state)
    reader.next_batch(7)
    state = reader.state_dict()
    archive_index, file_index = tmp_path / "archives.json", tmp_path / "files.json"
    write_json(archive_index, {"all_remote_sha256_verified": True, "archives": archives})
    write_json(file_index, inventory)
    checksum = json.loads((original / "manifest.json").read_text())["sha256"]
    return {"original": original, "state": state, "reader": reader, "sha": checksum,
            "archive_index": archive_index, "file_index": file_index, "archives": archives,
            "paths": paths, "members": members, "tmp": tmp_path}


def derived(corpus, **changes):
    options = {"target_tokens": 250, "expected_parent_sha256": corpus["sha"],
               "reader_state": corpus["state"]}
    options.update(changes)
    output = corpus["tmp"] / "mac-catalog"
    receipt = derive_catalog(corpus["original"], output, corpus["tmp"] / "mac-data", **options)
    return output, receipt


def stager(corpus, catalog):
    return ArchiveStager(catalog, corpus["archive_index"], corpus["file_index"], corpus["tmp"] / "cache",
                         fetch_archive=lambda item: corpus["paths"][item["path_in_repo"]], reserve_bytes=0)


def test_only_recognizable_local_appledouble_sidecars_are_ignored(corpus):
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    folder = cache.ensure_chunk('edu', 'a')
    sidecar = folder / '._train.bin'
    sidecar.write_bytes(bytes.fromhex('0005160700020000') + bytes(24))
    cache.ensure_chunk('edu', 'a')
    sidecar.write_bytes(b'not AppleDouble')
    with pytest.raises(ValueError, match='Unrecognized local metadata'):
        cache.ensure_chunk('edu', 'a')


def test_extra_frozen_source_does_not_change_training_recipe(corpus):
    output, _ = derived(corpus)
    inventory = json.loads(corpus['file_index'].read_text())
    inventory['initial'] = inventory['edu']
    write_json(corpus['file_index'], inventory)
    index = json.loads(corpus['archive_index'].read_text())
    item = copy.deepcopy(next(x for x in index['archives'] if '/edu/' in x['path_in_repo']))
    original_path = item['path_in_repo']
    item['path_in_repo'] = item['path_in_repo'].replace('/edu/', '/initial/')
    index['archives'].append(item)
    write_json(corpus['archive_index'], index)
    before = load_catalog(output)
    cache = ArchiveStager(output, corpus['archive_index'], corpus['file_index'], corpus['tmp'] / 'cache',
        fetch_archive=lambda item: corpus['paths'][original_path] if '/initial/' in item['path_in_repo']
            else corpus['paths'][item['path_in_repo']], reserve_bytes=0,
        extra_source_roots={'initial': corpus['tmp'] / 'initial-validation'})
    assert cache.estimate_chunks([('initial', 'a')])['download_bytes_needed'] == 0
    cache.stage_chunks([('initial', 'a')])
    assert load_catalog(output) == before and set(cache.catalog['sources']) == set(before['sources'])


def test_catalog_changes_only_paths_and_target_with_state_untouched(corpus):
    before = copy.deepcopy(corpus["state"])
    original = load_catalog(corpus["original"])
    output, receipt = derived(corpus)
    value = load_catalog(output)
    assert value["recipe"] == original["recipe"]
    assert value["recipe_id"] == before["recipe_id"]
    assert value["recipe"]["legacy"]["root"] == "/workspace/original/tokens"
    assert receipt["parent_catalog_sha256"] == corpus["sha"]
    assert value["target_tokens"] == 250
    for source in original["sources"]:
        assert value["sources"][source]["chunks"] == original["sources"][source]["chunks"]
        value["sources"][source]["directory"] = original["sources"][source]["directory"]
    value["target_tokens"] = original["target_tokens"]
    assert value == original
    assert corpus["state"] == before
    assert load_catalog(corpus["original"]) == original


@pytest.mark.parametrize("options", [
    {"target_tokens": 200}, {"target_tokens": 10000}, {"expected_parent_sha256": "0" * 64},
    {"target_tokens": True},
])
def test_invalid_derivation_refused_before_creating_output(corpus, options):
    with pytest.raises(ValueError):
        derived(corpus, **options)
    assert not (corpus["tmp"] / "mac-catalog").exists()


def test_state_inventory_change_refused_without_output(corpus):
    changed = copy.deepcopy(corpus["state"])
    changed["sources"]["edu"]["known"]["a"] = "0" * 64
    with pytest.raises(ValueError, match="inventory"):
        derived(corpus, reader_state=changed)
    assert not (corpus["tmp"] / "mac-catalog").exists()


def test_lazy_restore_preserves_exact_batches_and_resume_state(corpus):
    output, _ = derived(corpus)
    before = copy.deepcopy(corpus["state"])
    cache = stager(corpus, output)
    staged = cache.stage_current(before)
    assert set(staged) == set(before["sources"])
    assert before == corpus["state"]
    reader = cache.reader(before)
    np.testing.assert_array_equal(reader.next_batch(37), corpus["reader"].next_batch(37))
    assert reader.state_dict() == corpus["reader"].state_dict()
    saved = reader.state_dict()
    restarted = cache.reader(saved)
    np.testing.assert_array_equal(restarted.next_batch(31), reader.next_batch(31))
    assert restarted.state_dict() == reader.state_dict()


@pytest.mark.parametrize("defect", ["link", "traversal", "absolute", "duplicate", "extra", "missing", "payload"])
def test_unsafe_or_corrupt_archives_never_publish_chunk(corpus, defect):
    output, _ = derived(corpus)
    source = "edu"
    members = list(corpus["members"][source])
    if defect == "link":
        members[0] = (members[0][0], None)
    elif defect == "traversal":
        members.append(("../escape", b"bad"))
    elif defect == "absolute":
        members.append(("/tmp/escape", b"bad"))
    elif defect == "duplicate":
        members.append(members[0])
    elif defect == "extra":
        members.append(("a/unknown", b"bad"))
    elif defect == "missing":
        members.pop()
    elif defect == "payload":
        name, payload = members[0]
        members[0] = (name, b"X" * len(payload))
    relative = "datasets/tokenized-v1/edu/part-00000.tar.zst"
    archive = corpus["paths"][relative]
    compress_archive(archive, members)
    item = next(x for x in corpus["archives"] if x["path_in_repo"] == relative)
    item.update(bytes=archive.stat().st_size, sha256=sha(archive))
    write_json(corpus["archive_index"], {"all_remote_sha256_verified": True, "archives": corpus["archives"]})
    cache = stager(corpus, output)
    with pytest.raises(ValueError):
        cache.ensure_chunk("edu", "a")
    root = Path(load_catalog(output)["sources"]["edu"]["directory"])
    assert not (root / "a").exists()
    assert not (root / "b").exists()
    assert not list(root.glob(".mac-stage-*"))
    assert not (root.parent / "escape").exists()


def test_archive_checksum_and_existing_chunk_corruption_are_refused(corpus):
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    item = next(x for x in cache.archives.values() if "/edu/" in x["path_in_repo"])
    original = item["sha256"]
    item["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Compressed archive hash"):
        cache.ensure_chunk("edu", "a")
    item["sha256"] = original
    folder = cache.ensure_chunk("edu", "a")
    (folder / "train.bin").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash"):
        cache.ensure_chunk("edu", "a")
    assert (folder / "train.bin").read_bytes() == b"damaged"


def test_hf_archive_cache_symlink_is_resolved_but_output_links_are_refused(corpus):
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    relative = "datasets/tokenized-v1/edu/part-00000.tar.zst"
    original = corpus["paths"][relative]
    link = corpus["tmp"] / "hub-snapshot-archive"
    link.symlink_to(original)
    corpus["paths"][relative] = link
    folder = cache.ensure_chunk("edu", "a")
    train = folder / "train.bin"
    payload = train.read_bytes()
    train.unlink()
    elsewhere = corpus["tmp"] / "elsewhere.bin"
    elsewhere.write_bytes(payload)
    train.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="hash"):
        cache.ensure_chunk("edu", "a")


def test_low_disk_reserve_refuses_before_fetch(corpus):
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    cache.reserve_bytes = 10**30
    cache.fetch_archive = lambda _: pytest.fail("No download may start")
    with pytest.raises(ValueError, match="free disk"):
        cache.ensure_chunk("edu", "a")


def test_default_zero_transfer_budget_never_requests_network(corpus, monkeypatch):
    import huggingface_hub
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    monkeypatch.setattr(cache, "_local_archive", lambda _: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Network requested"))
    estimate = cache.estimate_current(corpus["state"])
    assert estimate["download_bytes_needed"] == sum(x["bytes"] for x in corpus["archives"])
    assert cache.download_bytes == 0
    with pytest.raises(ValueError, match="explicit budget"):
        cache.stage_current(corpus["state"])
    with pytest.raises(ValueError, match="remaining budget is 0"):
        cache.ensure_chunk("edu", "a")
    assert cache.download_bytes == 0


def test_aggregate_download_budget_is_checked_before_first_request(corpus, monkeypatch):
    import huggingface_hub
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    monkeypatch.setattr(cache, "_local_archive", lambda _: None)
    cache.max_download_bytes = sum(x["bytes"] for x in corpus["archives"]) - 1
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Partial acquisition started"))
    with pytest.raises(ValueError, match="explicit budget"):
        cache.stage_current(corpus["state"])
    assert cache.download_bytes == 0


def test_explicit_budget_counts_new_requests_and_pins_revision(corpus, monkeypatch):
    import huggingface_hub
    output, _ = derived(corpus)
    cache = stager(corpus, output)
    monkeypatch.setattr(cache, "_local_archive", lambda _: None)
    edu = next(x for x in corpus["archives"] if "/edu/" in x["path_in_repo"])
    cache.max_download_bytes = edu["bytes"]
    calls = []
    def local_fake(repo, filename, **options):
        calls.append((repo, filename, options))
        return corpus["paths"][filename]
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", local_fake)
    cache.ensure_chunk("edu", "a")
    assert cache.download_bytes == edu["bytes"]
    cache.ensure_chunk("edu", "b")  # Whole verified archive already staged; no second request.
    assert len(calls) == 1
    assert calls[0][0] == "keeeeenw/MicroLlama2"
    assert calls[0][2]["revision"] == "af9e5b1fa4455991d46aa2c4a623b905942b8053"
    assert calls[0][2]["token"] is False
    with pytest.raises(ValueError, match="remaining budget"):
        cache.ensure_chunk("dclm", "a")
    assert len(calls) == 1


def test_existing_archive_root_reused_without_transfer(corpus, monkeypatch):
    import huggingface_hub
    output, _ = derived(corpus)
    existing = corpus["tmp"] / "external-existing"
    for relative, path in corpus["paths"].items():
        target = existing / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)
    cache = ArchiveStager(output, corpus["archive_index"], corpus["file_index"], corpus["tmp"] / "cache",
                          existing_archive_roots=[existing], reserve_bytes=0)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("Unnecessary network request"))
    assert cache.estimate_current(corpus["state"])["download_bytes_needed"] == 0
    cache.stage_current(corpus["state"])
    assert cache.download_bytes == 0


def test_cli_default_writes_only_zero_budget_plan(corpus):
    import torch
    checkpoint = corpus["tmp"] / "training_state.pt"
    torch.save({"data": corpus["state"], "tokens": 100 + sum(corpus["state"]["source_targets"].values())}, checkpoint)
    output = corpus["tmp"] / "cli-plan"
    result = subprocess.run([sys.executable, "-m", "microllama.data_io.mac_staging",
        "--parent-catalog", str(corpus["original"]), "--parent-sha256", corpus["sha"],
        "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha(checkpoint),
        "--output", str(output), "--source-root", str(corpus["tmp"] / "cli-data"),
        "--cache-dir", str(corpus["tmp"] / "cli-cache"), "--archives", str(corpus["archive_index"]),
        "--files", str(corpus["file_index"]), "--target-tokens", "250"],
        capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["training_started"] is False
    assert report["target_authorization"] == "not established by data staging"
    assert report["transfer_plan"]["max_download_bytes"] == 0
    assert report["transfer_plan"]["download_bytes_already_requested"] == 0
    assert report["transfer_plan"]["download_bytes_needed"] == sum(x["bytes"] for x in corpus["archives"])
    assert not (corpus["tmp"] / "cli-data").exists()
    assert not (corpus["tmp"] / "cli-cache").exists()
    assert not (output / "staged-current.json").exists()
