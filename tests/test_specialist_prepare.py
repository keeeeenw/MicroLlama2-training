"""Bounded two-source admission, shared dedup and fail-closed acquisition."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from microllama.data_io import specialist_prepare as prep
from microllama.data_io.corpus import split_for
from microllama.data_io.download import sha256
from microllama.data_io.growth_prepare import DedupIndex, annotate


QUESTION = "Which process returns water from the surface of Earth to the atmosphere as vapor?"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return sha256(path)


def training_text(base):
    for n in range(1000):
        text = (base + " ") * 3000 + f" Example {n}. Literal <|im_start|> is ordinary text."
        if split_for(text) == "train":
            return text
    raise AssertionError("Could not find deterministic training split")


def cosmo(text, **extra):
    return {"text": text, "prompt": "Write a story about a community art exhibition.",
            "audience": "general", "format": "story", "seed_data": "web", **extra}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "require_space", lambda *a: 10**15)
    # The borrowed native pack helper has its own reserve check.
    monkeypatch.setattr("microllama.data_io.science_prepare.shutil.disk_usage", lambda *a: SimpleNamespace(free=10**15))
    legacy_dir = tmp_path / "GrowthData"; legacy_dir.mkdir()
    legacy = legacy_dir / "admission.sqlite3"
    index = DedupIndex(legacy)
    legacy_text = training_text("Ancient maritime navigation charts sailors coast compass")
    index.admit("legacy", annotate([{"id": "legacy", "text": legacy_text}])[0])
    index.db.commit(); index.db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); index.db.close()
    legacy_sha = sha256(legacy)
    tok_dir = tmp_path / "tokenizer"; tok_dir.mkdir()
    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "two": 2, "<|im_start|>": 3,
        "four": 4, "five": 5, "six": 6, "[EOS]": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken("<|im_start|>", special=True)])
    tok.save(str(tok_dir / "tokenizer.json"))
    write(tok_dir / "source_manifest.json", {"tokenizer_sha256": sha256(tok_dir / "tokenizer.json"),
        "eos_token_id": 7, "pad_token_id": 0})
    broad = training_text("Artistic village sculpture gallery paint sculpture music tradition")
    math = training_text("Integer algebra equation theorem derivative polynomial integration proof")
    raw_dir = tmp_path / "fixtures"; raw_dir.mkdir()
    files = {}
    records = {
        ("cosmopedia", 0): [cosmo(broad), cosmo(legacy_text), cosmo(broad, prompt=QUESTION), cosmo("")],
        ("cosmopedia", 1): [cosmo(training_text("Recipe cinnamon baking sourdough flour kitchen fermentation bread"))],
        ("finemath", 0): [{"text": broad, "int_score": 4}, {"text": broad.upper(), "int_score": 4}, {"text": math, "int_score": 4},
                           {"text": math + QUESTION, "int_score": 4}, {"text": math, "int_score": 3}],
        ("finemath", 1): [{"text": training_text("Probability permutation combinatorics distribution statistics counting"), "int_score": 5}],
    }
    sources = {}
    for name, (repo, config, count) in prep.SOURCE_SPECS.items():
        entries = [{"path": f"{config}/train-{i:05d}-of-{count:05d}.parquet", "bytes": 10**9,
                    "sha256": hashlib.sha256(f"{name}-{i}".encode()).hexdigest()} for i in range(count)]
        for n in (0, 1):
            path = raw_dir / f"{name}-{n}.parquet"
            pq.write_table(pa.Table.from_pylist(records[name, n]), path)
            entries[n].update(bytes=path.stat().st_size, sha256=sha256(path))
            files[entries[n]["path"]] = path
        sources[name] = {"repo": repo, "config": config, "revision": "a" * 40 if name == "cosmopedia" else "b" * 40,
                         "files": entries}
    manifest = tmp_path / "sources.json"; manifest_sha = write(manifest, {"sources": sources})
    benchmark = tmp_path / "benchmark.json"; benchmark_sha = write(benchmark, {"documents": [{"id": "heldout", "text": QUESTION}]})
    root = tmp_path / "SpecialistData"
    approval = tmp_path / "approval.json"
    options = prep.Options(root, manifest, manifest_sha, benchmark, benchmark_sha, legacy, tok_dir,
                           {"cosmopedia": 1, "finemath": 1}, approval, "0" * 64)
    calls = []

    def fetch(source, item, directory):
        calls.append(item["path"])
        path = directory / item["path"]; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[item["path"]], path)
        return path

    def approve(value, **extra):
        evidence = [{"path": str(Path(p).resolve()), "sha256": h} for p, h in value.prior_ledgers]
        digest = write(approval, {"scope": prep.APPROVAL_SCOPE, "output_root": str(root.resolve()),
            "source_manifest_sha256": value.source_manifest_sha256, "benchmark_documents_sha256": value.benchmark_sha256,
            "raw_cap_bytes": value.raw_cap_bytes, "target_valid_targets": value.target_valid_targets,
            "prior_ledgers": evidence, "authorization": "Direct user request to prepare this four-source mix", **extra})
        return replace(value, cap_approval_sha256=digest)

    return SimpleNamespace(options=approve(options), approve=approve, fetch=fetch, calls=calls, root=root,
                           sources=sources, legacy=legacy, legacy_sha=legacy_sha, files=files,
                           broad=broad, math=math, tmp=tmp_path)


def test_broad_cosmo_and_math_shared_dedup_native_packing(fixture):
    f = fixture
    state = prep.run_preparation(f.options, fetch=f.fetch)
    assert state["phase"] == "complete" and len(f.calls) == 2
    assert {i["source"] for i in state["selected"]} == {"cosmopedia", "finemath"}
    first = prep.read_json(f.root / "receipts/sp000001.json")
    second = prep.read_json(f.root / "receipts/sp000002.json")
    assert first["counts"] == {"accepted": 1, "legacy_exact_overlap": 1, "benchmark_overlap": 1, "empty_or_malformed_text": 1}
    assert second["counts"] == {"exact_overlap": 1, "near_overlap": 1, "accepted": 1, "benchmark_overlap": 1, "score_below_four_or_malformed": 1}
    assert sha256(f.legacy) == f.legacy_sha
    for source in state["chunks"]:
        assert state["valid_targets"][source] >= 8192
        for name in state["chunks"][source]:
            directory = f.root / "sources" / source / "tokens" / name
            ids = np.fromfile(directory / "train.bin", dtype="<u2")
            assert 3 not in ids and ids[-1] == 7
            meta = prep.read_json(directory / "manifest.json")
            assert meta["sequence_length"] == 8192
            assert meta["source_name"] == source
            assert meta["admission_policy_digest"] == prep.POLICY_DIGEST
    samples = prep.read_json(f.root / "receipts/sp000001.samples.json")
    assert samples["accepted"][0]["format"] == "story"  # Broad corpus has no science classifier.


def test_completed_verify_without_repeat_and_corruption_blocks(fixture):
    f = fixture
    first = prep.run_preparation(f.options, fetch=f.fetch)
    before = list(f.calls)
    assert prep.run_preparation(f.options, fetch=f.fetch) == first
    assert f.calls == before
    name = next(iter(first["chunks"]["finemath"]))
    binary = f.root / "sources/finemath/tokens" / name / "train.bin"
    binary.write_bytes(binary.read_bytes() + b"xx")
    with pytest.raises(ValueError, match="size"):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert f.calls == before


@pytest.mark.parametrize("sidecar", ["samples.json", "decisions.jsonl"])
def test_published_audit_sidecar_corruption_blocks_repeat(fixture, sidecar):
    f = fixture
    prep.run_preparation(f.options, fetch=f.fetch)
    path = f.root / "receipts" / ("sp000001." + sidecar)
    path.write_bytes(path.read_bytes() + b" ")
    before = list(f.calls)
    with pytest.raises(ValueError, match="Checksum"):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert f.calls == before


def test_pilot_stops_and_bulk_needs_bound_review(fixture):
    f = fixture
    sources = f.sources
    cap = sum(sources[s]["files"][0]["bytes"] for s in sources)
    options = f.approve(replace(f.options, target_valid_targets={s: 10**9 for s in sources}, pilot_cap_bytes=cap))
    state = prep.run_preparation(options, fetch=f.fetch)
    assert state["phase"] == "paused" and state["stop_reason"] == "pilot_raw_cap"
    assert {i["source"] for i in state["selected"]} == set(sources)
    assert state["raw_bytes_selected"] <= cap and len(f.calls) == 2
    with pytest.raises(ValueError, match="manual review"):
        prep.run_preparation(replace(options, stage="continue"), fetch=f.fetch)
    assert len(f.calls) == 2
    write(f.root / "pilot-review.json", {"phase": "passed", "prepare_state_sha256": "0" * 64,
                                        "manual_samples_reviewed": True})
    with pytest.raises(ValueError, match="manual review"):
        prep.run_preparation(replace(options, stage="continue"), fetch=f.fetch)
    assert len(f.calls) == 2


def test_reviewed_bulk_preserves_pilot_and_later_stops_at_same_cap(fixture):
    f = fixture
    cap = sum(f.sources[s]["files"][0]["bytes"] for s in f.sources)
    total = sum(f.sources[s]["files"][i]["bytes"] for s in f.sources for i in (0, 1))
    options = f.approve(replace(f.options, target_valid_targets={s: 10**9 for s in f.sources},
                               pilot_cap_bytes=cap, raw_cap_bytes=total))
    pilot = prep.run_preparation(options, fetch=f.fetch)
    before = {s: dict(c) for s, c in pilot["chunks"].items()}
    state_path = f.root / "prepare-state.json"
    write(f.root / "pilot-review.json", {"phase": "passed", "prepare_state_sha256": sha256(state_path),
          "manual_samples_reviewed": True, "sources_reviewed": ["cosmopedia", "finemath"]})
    result = prep.run_preparation(replace(options, stage="continue"), fetch=f.fetch)
    assert result["phase"] == "paused" and result["stop_reason"] == "raw_cap_exhausted"
    assert len(f.calls) == 4 and result["cumulative_raw_bytes_selected"] == total
    assert all(result["chunks"][s][n] == c for s in before for n, c in before[s].items())
    assert all(item["batch"] == 1 for item in result["selected"][2:])
    review = f.root / "pilot-review.json"
    changed = prep.read_json(review); changed["manual_samples_reviewed"] = False; write(review, changed)
    with pytest.raises(ValueError, match="Checksum"):
        prep.run_preparation(replace(options, stage="continue"), fetch=f.fetch)
    assert len(f.calls) == 4


def test_explicit_direct_bulk_preserves_completed_data_and_defers_review(fixture):
    f = fixture
    cap = sum(f.sources[s]["files"][0]["bytes"] for s in f.sources)
    total = sum(f.sources[s]["files"][i]["bytes"] for s in f.sources for i in (0, 1))
    options = f.approve(replace(f.options, target_valid_targets={s: 10**9 for s in f.sources},
                               pilot_cap_bytes=cap, raw_cap_bytes=total))
    pilot = prep.run_preparation(options, fetch=f.fetch)
    state_path = f.root / "prepare-state.json"
    approval_path = f.tmp / "direct-bulk.json"
    value = {"scope": "user-approved-direct-specialist-bulk-v1", "output_root": str(f.root.resolve()),
             "prepare_state_sha256": sha256(state_path), "preparation_pin_digest": prep.digest(pilot["pin"]),
             "review_timing": "after-full-preparation-before-training", "authorization": "Explicit user request"}
    bulk = replace(options, stage="continue", admission_workers=2, bulk_authorization=approval_path,
                   bulk_authorization_sha256=write(approval_path, value))
    for field in ("prepare_state_sha256", "preparation_pin_digest", "scope", "review_timing", "output_root"):
        invalid = dict(value); invalid[field] = "wrong"
        bad = replace(bulk, bulk_authorization_sha256=write(approval_path, invalid))
        with pytest.raises(ValueError, match="Direct bulk authorization"):
            prep.run_preparation(bad, fetch=f.fetch)
        assert len(f.calls) == 2
    bulk = replace(bulk, bulk_authorization_sha256=write(approval_path, value))
    result = prep.run_preparation(bulk, fetch=f.fetch)
    assert len(f.calls) == 4 and result["stop_reason"] == "raw_cap_exhausted"
    assert not (f.root / "pilot-review.json").exists()
    assert result["direct_bulk_authorization"]["review_timing"] == "after-full-preparation-before-training"
    assert all(result["chunks"][s][n] == c for s, chunks in pilot["chunks"].items() for n, c in chunks.items())
    value["authorization"] = "changed"
    write(approval_path, value)
    with pytest.raises(ValueError, match="Checksum"):
        prep.run_preparation(bulk, fetch=f.fetch)
    assert len(f.calls) == 4


@pytest.mark.parametrize("stage,path,checksum", [("pilot", True, True), ("continue", True, False),
                                                ("continue", False, True)])
def test_partial_or_wrong_stage_bulk_authorization_blocks_before_acquisition(fixture, stage, path, checksum):
    f = fixture
    with pytest.raises(ValueError, match="Direct bulk authorization needs"):
        prep.run_preparation(replace(f.options, stage=stage,
            bulk_authorization=f.tmp / "direct.json" if path else None,
            bulk_authorization_sha256="0" * 64 if checksum else None), fetch=f.fetch)
    assert not f.calls


def prior_ledger(f, directory, number=0):
    source = f.sources["cosmopedia"]
    item = {**source["files"][number], "sequence": 1, "batch": 0}
    path = f.tmp / directory / "prepare-state.json"
    identity = write(path, {"phase": "paused", "pin": {k: source[k] for k in ("repo", "revision", "config")},
                            "selected": [item], "raw_bytes_selected": item["bytes"], "error": None})
    return path, identity


def test_prior_union_charges_once_skips_quarantined_shards(fixture):
    f = fixture
    prior = (prior_ledger(f, "old-science"), prior_ledger(f, "v2-science"))
    options = f.approve(replace(f.options, prior_ledgers=prior))
    state = prep.run_preparation(options, fetch=f.fetch)
    used = f.sources["cosmopedia"]["files"][0]["bytes"]
    assert state["pin"]["prior_unique_raw_bytes"] == used
    assert state["cumulative_raw_bytes_selected"] == used + state["raw_bytes_selected"]
    assert f.sources["cosmopedia"]["files"][0]["path"] not in f.calls
    assert state["selected"][0]["path"] == f.sources["cosmopedia"]["files"][1]["path"]


def test_actual_legacy_science_schema_without_aggregate_uses_selected_sum(fixture):
    f = fixture
    path, _ = prior_ledger(f, "old-science")
    value = prep.read_json(path)
    value.pop("raw_bytes_selected")
    value["pin"].update(preparation_version="cosmopedia-science-prepare-v2",
                        raw_accounting={"prior_raw_bytes": 12345})
    identity = write(path, value)
    options = f.approve(replace(f.options, prior_ledgers=((path, identity),)))
    result = prep.run_preparation(options, fetch=f.fetch)
    assert result["pin"]["prior_unique_raw_bytes"] == f.sources["cosmopedia"]["files"][0]["bytes"]
    assert result["cumulative_raw_bytes_selected"] == result["raw_bytes_selected"] + result["pin"]["prior_unique_raw_bytes"]


def test_shared_admission_database_cannot_disappear_between_stages(fixture):
    f = fixture
    prep.run_preparation(f.options, fetch=f.fetch)
    (f.root / "admission.sqlite3").unlink()
    before = list(f.calls)
    with pytest.raises(FileNotFoundError):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert f.calls == before


def test_prior_producer_must_be_stopped_and_locked(fixture):
    f = fixture
    path, identity = prior_ledger(f, "old-science")
    value = prep.read_json(path); value["phase"] = "acquiring"; identity = write(path, value)
    options = f.approve(replace(f.options, prior_ledgers=((path, identity),)))
    with pytest.raises(ValueError, match="unambiguous"):
        prep.run_preparation(options, fetch=f.fetch)
    assert not f.calls
    value["phase"] = "paused"; identity = write(path, value)
    options = f.approve(replace(f.options, prior_ledgers=((path, identity),)))
    import fcntl
    with (path.parent / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            prep.run_preparation(options, fetch=f.fetch)
    assert not f.calls


@pytest.mark.parametrize("phase", ["acquiring", "filtering", "packing", "published", "failed"])
def test_uncertain_or_failed_attempt_never_relaunches(fixture, phase):
    f = fixture
    prep.run_preparation(f.options, fetch=f.fetch)
    state_path = f.root / "prepare-state.json"
    state = prep.read_json(state_path); state["phase"] = phase; write(state_path, state)
    before = list(f.calls)
    with pytest.raises(RuntimeError, match="uncertain"):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert f.calls == before


def test_wrong_source_approval_budget_and_tokenizer_block_before_download(fixture):
    f = fixture
    for option in (replace(f.options, raw_cap_bytes=prep.RAW_CAP + 1),
                   replace(f.options, pilot_cap_bytes=prep.PILOT_CAP + 1),
                   replace(f.options, batch_cap_bytes=prep.BATCH_CAP + 1)):
        with pytest.raises(ValueError, match="bounds"):
            prep.run_preparation(option, fetch=f.fetch)
    option = f.approve(f.options, scope="cosmopedia-science-preparation-v1")
    with pytest.raises(ValueError, match="approval"):
        prep.run_preparation(option, fetch=f.fetch)
    assert not f.calls
    f.approve(f.options)
    manifest = prep.read_json(f.options.source_manifest)
    manifest["sources"]["finemath"]["files"].pop()
    changed = write(f.options.source_manifest, manifest)
    with pytest.raises(ValueError, match="All official"):
        prep.run_preparation(replace(f.options, source_manifest_sha256=changed), fetch=f.fetch)
    assert not f.calls


def test_raw_checksum_failure_is_durable_and_never_retried(fixture):
    f = fixture
    def corrupt(source, item, directory):
        path = f.fetch(source, item, directory)
        path.write_bytes(b"x" * path.stat().st_size)
        return path
    with pytest.raises(ValueError, match="Checksum"):
        prep.run_preparation(f.options, fetch=corrupt)
    state = prep.read_json(f.root / "prepare-state.json")
    assert state["phase"] == "failed" and len(state["selected"]) == 1
    assert state["raw_bytes_selected"] == state["selected"][0]["bytes"]
    with pytest.raises(RuntimeError, match="uncertain"):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert len(f.calls) == 1


def test_reserve_and_root_escape_block_without_publication(fixture, monkeypatch):
    f = fixture
    def no_space(*a):
        raise RuntimeError("Storage reserve would be crossed")
    monkeypatch.setattr(prep, "require_space", no_space)
    with pytest.raises(RuntimeError, match="reserve"):
        prep.run_preparation(f.options, fetch=f.fetch)
    assert not f.calls and not list(f.root.glob("sources/*/tokens/*/manifest.json"))


def test_math_adapter_requires_exact_integer_score_and_broad_text_only():
    assert prep.quality_reason({"text": "math", "int_score": 4}, "finemath") is None
    assert prep.quality_reason({"text": "math", "int_score": 4.0}, "finemath") is not None
    assert prep.quality_reason({"text": "math", "int_score": True}, "finemath") is not None
    assert prep.quality_reason(cosmo("Fiction and humanities are intentionally admitted broadly."), "cosmopedia") is None


def test_parallel_screening_preserves_ordered_admission_and_packed_payloads(fixture):
    f = fixture
    serial = prep.run_preparation(f.options, fetch=f.fetch)
    parallel_root = f.tmp / "ParallelSpecialistData"
    approval = prep.read_json(f.options.cap_approval)
    approval["output_root"] = str(parallel_root.resolve())
    approval_path = f.tmp / "parallel-approval.json"
    options = replace(f.options, output_root=parallel_root, cap_approval=approval_path,
                      cap_approval_sha256=write(approval_path, approval), admission_workers=2,
                      tokenizer_threads=4)
    parallel = prep.run_preparation(options, fetch=f.fetch)
    assert parallel["selected"] == serial["selected"]
    assert parallel["valid_targets"] == serial["valid_targets"]
    assert sha256(f.legacy) == f.legacy_sha
    for number in range(1, len(serial["selected"]) + 1):
        name = f"sp{number:06d}"
        before = prep.read_json(f.root / "receipts" / (name + ".json"))
        after = prep.read_json(parallel_root / "receipts" / (name + ".json"))
        for field in ("counts", "filtered_sha256", "decisions_sha256", "samples_sha256", "valid_targets"):
            assert after[field] == before[field]
    for source, chunks in serial["chunks"].items():
        assert set(parallel["chunks"][source]) == set(chunks)
        for name in chunks:
            for filename in ("train.bin", "validation.bin", "train.documents.jsonl", "validation.documents.jsonl"):
                assert sha256(f.root / "sources" / source / "tokens" / name / filename) == sha256(
                    parallel_root / "sources" / source / "tokens" / name / filename)


def test_parallel_screening_preserves_order_across_prefetched_batches(fixture):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    f = fixture
    records = [cosmo(f"Distinct document number {n} includes ordinary literary discussion.") for n in range(600)]
    raw = f.tmp / "order.parquet"
    pq.write_table(pa.Table.from_pylist(records), raw)
    item = {"source": "cosmopedia", "path": "ordered.parquet"}
    legacy = prep.ReadOnlyAdmission(f.legacy)
    blocker = prep.BenchmarkOverlapBlocker([QUESTION])
    try:
        expected = list(prep.screened_rows(raw, item, legacy, blocker))
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"),
                initializer=prep.initialize_screen_worker,
                initargs=(str(f.options.benchmark_documents), f.options.benchmark_sha256,
                          str(f.legacy), legacy.identity)) as pool:
            actual = list(prep.screened_rows(raw, item, legacy, blocker, pool))
        assert actual == expected
        assert [row[1] for row in actual] == [f"cosmopedia:ordered.parquet:{n}" for n in range(600)]
    finally:
        legacy.close()


@pytest.mark.parametrize("workers,threads", [(0, 2), (33, 2), (True, 2), (2, 17)])
def test_parallel_cpu_settings_are_bounded_before_acquisition(fixture, workers, threads):
    f = fixture
    with pytest.raises(ValueError, match="CPU packing"):
        prep.run_preparation(replace(f.options, admission_workers=workers, tokenizer_threads=threads), fetch=f.fetch)
    assert not f.calls
