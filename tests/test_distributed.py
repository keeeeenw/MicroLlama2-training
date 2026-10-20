"""Actual multiprocess gradient/restart parity, without requiring CUDA hardware."""

import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from microllama.training.distributed import (
    LossModel, accumulate_gradients, local_accumulation, local_rows,
    inspect_corpus, enforce_coverage, restart_rng, restore_rng, verify_checkpoint, fixed_validation_config,
)
from microllama.training.cuda import build_model
from microllama.training.objective import chunked_loss, target_mask, normalize_accumulated_gradients
from microllama.training.optimizers import build_optimizer
from scripts.cloud.run_vast import main as launch, export_checkpoint_manifest, checkpoint_free_bytes
from scripts.cloud.hardware import estimate
from test_engram import TINY, tiny_engram


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    dist.init_process_group("gloo", rank=rank, world_size=2,
        init_method="file://" + str(Path(directory) / "rendezvous"), timeout=timedelta(seconds=60))
    try:
        for revised, micro_batch in ((False, 1), (False, 2), (True, 1), (True, 2)):
            torch.manual_seed(43)
            model = tiny_engram(revised=True) if revised else build_model(TINY, gradient_checkpointing=False)
            reference = copy.deepcopy(model)
            opt = build_optimizer(model, memory_lr_multiplier=5 if revised else 1)
            ref_opt = build_optimizer(reference, memory_lr_multiplier=5 if revised else 1)
            wrapped = DDP(LossModel(model, 3), gradient_as_bucket_view=True, broadcast_buffers=False)
            # Deliberately unequal valid-target counts, including EOS and PAD.
            global_rows = np.array([[2,3,7,4,5,0,0,0,0], [9,10,11,12,13,14,15,16,17],
                                    [2,7,4,7,5,7,6,8,9], [20,21,22,23,24,25,26,27,28]], dtype=np.int64)
            rows_per_batch = 2 * micro_batch
            batches = [local_rows(global_rows[i:i+rows_per_batch], rank, 2, micro_batch)
                       for i in range(0, len(global_rows), rows_per_batch)]
            opt.zero_grad(set_to_none=True)
            loss, count = accumulate_gradients(wrapped, model, batches, device)
            ref_opt.zero_grad(set_to_none=True)
            tensor = torch.tensor(global_rows)
            mask = target_mask(tensor[:,:-1], tensor[:,1:],7,0)
            reference_loss = chunked_loss(reference, tensor[:,:-1], tensor[:,1:],3,loss_mask=mask,reduction="sum")
            reference_loss.backward()
            normalize_accumulated_gradients(reference,int(mask.sum()))
            assert count == int(mask.sum())
            assert abs(loss - float(reference_loss)/count) < 1e-6
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=3e-7,rtol=3e-4)
            opt.step(); ref_opt.step()
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,atol=2e-6,rtol=3e-4)
            # Each rank has its own RNG state; restoration reproduces its next draws.
            torch.manual_seed(100 + rank)
            rng = restart_rng(device)
            expected = torch.rand(5)
            restore_rng(rng,device)
            torch.testing.assert_close(torch.rand(5),expected,atol=0,rtol=0)
            snapshot = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(opt.state_dict())
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                accumulate_gradients(wrapped,model,batches,device)
                opt.step()
                if _ == 0:
                    expected_next = copy.deepcopy(model.state_dict())
                    model.load_state_dict(snapshot)
                    opt.load_state_dict(copy.deepcopy(optimizer_state))
                else:
                    for key,value in model.state_dict().items():
                        torch.testing.assert_close(value,expected_next[key],atol=0,rtol=0)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"passed":True}))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_ddp_matches_global_batch_and_restores_next_update(tmp_path):
    code = "import sys,torch.multiprocessing as mp; from test_distributed import ddp_worker; mp.spawn(ddp_worker,args=(sys.argv[1],),nprocs=2,join=True)"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "GLOO_SOCKET_IFNAME":"lo0" if sys.platform == "darwin" else "lo",
           "PYTHONPATH":str(Path(__file__).resolve().parent) + os.pathsep + str(Path.cwd())}
    result = subprocess.run([sys.executable,"-c",code,str(tmp_path)],env=env,capture_output=True,text=True,timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(json.loads((tmp_path/f"rank-{rank}.json").read_text())["passed"] for rank in (0,1))


@pytest.mark.parametrize("micro_batch,global_accumulation", [(1, 16), (2, 8)])
def test_rank_batches_preserve_single_gpu_global_cursor(tmp_path, micro_batch, global_accumulation):
    from microllama.data_io.blocks import GrowingTokenBlocks
    from test_streaming import summary
    from test_data import write_shard
    write_shard(tmp_path / "a", range(200))
    summary(tmp_path,True)
    a,b = [GrowingTokenBlocks(tmp_path,8) for _ in range(2)]
    expected = a.next_batch(16)
    groups = [b.next_batch(4 * micro_batch) for _ in range(local_accumulation(global_accumulation,4))]
    np.testing.assert_array_equal(np.concatenate(groups),expected)
    assert a.state_dict() == b.state_dict()
    for group in groups:
        np.testing.assert_array_equal(np.concatenate([local_rows(group,r,4,micro_batch) for r in range(4)]),group)
    with pytest.raises(ValueError):
        local_accumulation(3,4)


def test_training_microbatch_does_not_change_frozen_validation_blocks(tmp_path):
    from microllama.data_io.blocks import TokenBlocks
    from test_data import write_shard
    # Both profiles visit exactly the same 128 validation rows, in the same order.
    write_shard(tmp_path / "one", range(1200))
    configs = [fixed_validation_config({"micro_batch_size": micro, "eval_batches": 128}) for micro in (1, 2)]
    readers = [TokenBlocks(tmp_path, 8, split="validation", seed=42) for _ in configs]
    batches = [np.concatenate([reader.next_batch(config["micro_batch_size"])
               for _ in range(config["eval_batches"])]) for reader, config in zip(readers, configs)]
    np.testing.assert_array_equal(batches[0], batches[1])
    assert len(batches[0]) == 128


def test_inventory_boundary_counts_and_same_length_corruption(tmp_path):
    folder = tmp_path / "one"; folder.mkdir()
    values = np.array([2,3,7,4,5,0,8,9,10],dtype="<u2")
    meta = {"source_revision":"rev","tokenizer_sha256":"tok","splits":{}}
    for split in ("train","validation"):
        binary=folder/f"{split}.bin"; binary.write_bytes(values.tobytes())
        meta["splits"][split]={"tokens":len(values),"sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
    (folder/"manifest.json").write_text(json.dumps(meta))
    (tmp_path/"manifest.json").write_text(json.dumps({"complete":True,"source_revision":"rev","tokenizer_sha256":"tok","train_tokens":9}))
    result=inspect_corpus(tmp_path,4,"tok")
    assert result["valid_targets_per_pass"] == 5
    (folder/"train.bin").write_bytes(b"x"*len(values)*2)
    with pytest.raises(ValueError,match="Corrupt"):
        inspect_corpus(tmp_path,4,"tok")


def test_50b_cannot_repeat_the_10bt_sample():
    inventory={"valid_targets_per_pass":9_815_000_000}
    assert enforce_coverage(10_000_000_000,inventory)["nominal_tail_replay_targets"] == 185_000_000
    with pytest.raises(ValueError,match="more data"):
        enforce_coverage(50_000_000_000,inventory)
    assert enforce_coverage(50_000_000_000,{"valid_targets_per_pass":51_000_000_000})["nominal_tail_replay_targets"] == 0


def test_continuation_disk_budget_covers_three_full_publications_and_reserve(tmp_path):
    for name, size in [("model.safetensors", 100), ("training_state.pt", 200)]:
        (tmp_path / name).write_bytes(b"x" * size)
    required = checkpoint_free_bytes(tmp_path)
    assert required >= 32 * 2**30 + 3 * 300
    assert required == 32 * 2**30 + 990
    assert checkpoint_free_bytes() == 56 * 2**30
    # A qualification checkpoint already occupies one retained slot. On resume,
    # reserve only the additional slots rather than counting its bytes twice.
    assert checkpoint_free_bytes(tmp_path, retained=1) == 32 * 2**30 + 660
    assert checkpoint_free_bytes(tmp_path, retained=2) == 32 * 2**30 + 330
    assert checkpoint_free_bytes(tmp_path, keep=4, retained=2) == 32 * 2**30 + 990
    assert checkpoint_free_bytes(keep=4) == 72 * 2**30


def test_plan_defaults_dense_and_is_side_effect_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys,"argv",["run_vast","--plan","--data-dir",str(tmp_path/"new")])
    launch()
    plan=json.loads(capsys.readouterr().out)
    assert plan["config"]["target_tokens"] == 10_000_000_000
    assert plan["per_rank_accumulation"] == 4 and "--engram" not in plan["command"]
    assert not (tmp_path/"new").exists()
    monkeypatch.setattr(sys,"argv",["run_vast","--plan","--engram","--target-tokens","50000000000","--data-dir",str(tmp_path/"new")])
    launch()
    plan=json.loads(capsys.readouterr().out)
    assert "--engram" in plan["command"] and plan["config"]["target_tokens"] == 50_000_000_000
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("profile,micro,accumulation,environment", [
    ("4090-24", 1, 4, ".venv-train"),
    ("4090-48", 2, 2, ".venv-train"),
    ("5090-32", 1, 4, ".venv-train-blackwell"),
])
def test_cloud_profiles_preserve_recipe_and_budget_without_launching(tmp_path, monkeypatch, capsys,
                                                                  profile, micro, accumulation, environment):
    monkeypatch.setattr(sys, "argv", ["run_vast", "--plan", "--hardware-profile", profile,
        "--qualification-updates", "40", "--target-tokens", "50000000000", "--data-dir", str(tmp_path / "new")])
    launch()
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"]["schedule"] == "constant-after-warmup-v1"
    assert plan["config"]["target_tokens"] == 50_000_000_000
    assert plan["config"]["micro_batch_size"] == micro
    assert plan["per_rank_accumulation"] == accumulation
    assert plan["global_input_positions_per_update"] == 131072
    assert plan["validation_micro_batch_size"] == 1
    assert environment + "/bin/python" in plan["command"][0]
    assert "--nproc-per-node=4" in plan["command"] and "--engram" not in plan["command"]
    assert plan["command"][plan["command"].index("--max-steps") + 1] == "40"
    assert not (tmp_path / "new").exists()


def test_cloud_qualification_counts_new_updates_and_resume_preserves_partition(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    config = json.loads(Path("configs/train-8k-local.json").read_text())
    (checkpoint / "run_config.json").write_text(json.dumps(config))
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"step": 8000}))
    common = ["run_vast", "--plan", "--hardware-profile", "4090-48", "--engram",
              "--qualification-updates", "40", "--data-dir", str(tmp_path / "new")]
    monkeypatch.setattr(sys, "argv", common + ["--continue-from", str(checkpoint)])
    launch()
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"][plan["command"].index("--max-steps") + 1] == "8040"
    assert plan["config"]["micro_batch_size"] == 2 and "--engram" in plan["command"]
    monkeypatch.setattr(sys, "argv", common + ["--resume", str(checkpoint)])
    launch()
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"] == config  # Hardware profile never rewrites exact-resume state.
    monkeypatch.setattr(sys, "argv", common + ["--resume", str(checkpoint), "--micro-batch-size", "2"])
    with pytest.raises(ValueError, match="Exact resume"):
        launch()
    monkeypatch.setattr(sys, "argv", common + ["--micro-batch-size", "1"])
    launch()
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"]["micro_batch_size"] == 1 and plan["per_rank_accumulation"] == 4
    assert not (tmp_path / "new").exists()


def test_cloud_profile_rejects_wrong_card_count(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_vast", "--plan", "--hardware-profile", "5090-32",
                                      "--gpus", "1", "--data-dir", str(tmp_path / "new")])
    with pytest.raises(SystemExit):
        launch()
    assert not (tmp_path / "new").exists()


def test_runtime_cost_estimates_use_remaining_targets_and_whole_instance_price():
    result = estimate(10_000_000_000, already_trained=1_000_000_000,
                      hourly_prices=[2, 2.2, 3], measured_rates=[60_000, 66_000, 100_000])
    assert result["remaining_tokens"] == 9_000_000_000
    a, b, c = result["profiles"]
    assert a["hours"] == pytest.approx(41.6666666667)
    assert a["compute_cost"] == pytest.approx(83.3333333333)
    assert b["compute_cost"] == pytest.approx(a["compute_cost"])
    assert c["compute_cost"] < a["compute_cost"]
    assert b["break_even_hourly_price_ratio_to_4090_24"] == pytest.approx(1.1)
    assert c["break_even_hourly_price_ratio_to_4090_24"] == pytest.approx(5 / 3)
    for row in result["profiles"]:
        assert row["hours_range"] == [row["hours"], row["hours"]]
        assert row["basis"].startswith("user-supplied")
    forecasts = estimate(50_000_000_000)["profiles"]
    assert all(row["hours_range"][0] < row["hours"] < row["hours_range"][1] for row in forecasts)
    assert all(row["compute_cost"] is None for row in forecasts)
    for bad in ([1, 2], [1, float("nan"), 3], [1, -1, 3], [1, float("inf"), 3]):
        with pytest.raises(ValueError):
            estimate(10_000_000_000, hourly_prices=bad)
    with pytest.raises(ValueError):
        estimate(10, already_trained=10)
    single = estimate(50_000_000_000)["current_single_4090"]
    assert single["hours"] / 24 == pytest.approx(29.6771120608)
    assert single["hours_range"][0] < single["hours"] < single["hours_range"][1]
    single = estimate(10_000_000_000, single_hourly_price=1, single_measured_rate=20_000)["current_single_4090"]
    assert single["compute_cost"] == pytest.approx(138.8888888889)
    with pytest.raises(ValueError):
        estimate(10, single_measured_rate=float("nan"))


def test_estimator_cli_is_side_effect_free_and_returns_both_budgets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_vast", "--estimate", "--hourly-prices", "2", "2.2", "3",
                                      "--data-dir", str(tmp_path / "new")])
    launch()
    output = json.loads(capsys.readouterr().out)
    assert [row["target_tokens"] for row in output] == [10_000_000_000, 50_000_000_000]
    assert output[0]["profiles"][0]["compute_cost"] == pytest.approx(92.5925925926)
    assert output[0]["current_single_4090"]["hours"] == pytest.approx(142.4501424501)
    assert not (tmp_path / "new").exists()


def test_blackwell_setup_keeps_live_environment_untouched(tmp_path, monkeypatch):
    import scripts.cloud.run_vast as launcher
    from scripts.cloud.hardware import profiles
    commands = []
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(launcher, "run", lambda command, env: commands.append([str(v) for v in command]))
    launcher.setup({}, profiles()["5090-32"])
    assert any("torch==2.9.1" in command and "https://download.pytorch.org/whl/cu128" in command
               for command in commands)
    assert any(str(tmp_path / "requirements-train-blackwell.lock") in command for command in commands)
    assert any(str(tmp_path / ".venv-train-blackwell/bin/python") in command for command in commands)
    assert not any(str(tmp_path / ".venv-train/bin/python") in command for command in commands)


@pytest.mark.parametrize("profile,name,memory,capability,cuda_version,arches,bf16,error", [
    ("4090-48", "NVIDIA GeForce RTX 4090", 24, (8, 9), "12.1", ["sm_89"], True, "does not match"),
    ("5090-32", "NVIDIA GeForce RTX 5090", 32, (12, 0), "12.1", ["sm_89"], True, "Blackwell requires"),
    ("5090-32", "NVIDIA GeForce RTX 5090", 32, (12, 0), "12.8", ["sm_90"], True, "Blackwell requires"),
    ("4090-24", "NVIDIA GeForce RTX 4090", 24, (8, 9), "12.1", ["sm_89"], False, "BF16"),
])
def test_gpu_preflight_rejects_bad_capacity_or_wheel_before_allocating(monkeypatch, profile, name,
        memory, capability, cuda_version, arches, bf16, error):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts.cloud.hardware import cuda_preflight
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index:
                        SimpleNamespace(name=name, total_memory=memory * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arches)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: pytest.fail("Must reject before allocating"))
    with pytest.raises(RuntimeError, match=error):
        cuda_preflight(4, profile)


def test_verified_checkpoint_publication_rejects_modified_payload(tmp_path):
    file=tmp_path/"model.safetensors"; file.write_bytes(b"weights")
    files={"model.safetensors":{"bytes":7,"sha256":hashlib.sha256(b"weights").hexdigest()}}
    for name in ("training_state.pt","config.json","run_config.json","tokenizer.json"):
        (tmp_path/name).write_bytes(b"fixture")
        files[name]={"bytes":7,"sha256":hashlib.sha256(b"fixture").hexdigest()}
    (tmp_path/"checkpoint_manifest.json").write_text(json.dumps({"schema_version":1,"files":files}))
    assert len(verify_checkpoint(tmp_path)) == 64
    file.write_bytes(b"changes")
    with pytest.raises(ValueError,match="verification"):
        verify_checkpoint(tmp_path)


def test_resume_plan_preserves_saved_target_name_and_no_writes(tmp_path, monkeypatch, capsys):
    checkpoint=tmp_path/"checkpoint"; checkpoint.mkdir()
    config=json.loads(Path("configs/train-8k-v2.json").read_text())
    config.update(run_name="custom-cloud-50b",target_tokens=50_000_000_000,eval_batches=128)
    (checkpoint/"run_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys,"argv",["run_vast","--plan","--resume",str(checkpoint),"--data-dir",str(tmp_path/"new")])
    launch()
    plan=json.loads(capsys.readouterr().out)
    assert plan["config"] == config
    assert not (tmp_path/"new").exists()


def test_safe_run_names_reject_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(sys,"argv",["run_vast","--plan","--run-name","../other","--data-dir",str(tmp_path/"new")])
    with pytest.raises(ValueError,match="safe directory"):
        launch()
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("setup_flags", [[], ["--setup", "--prepare-only"]])
def test_launch_refuses_gpu_owned_by_existing_experiment(tmp_path, monkeypatch, setup_flags):
    import fcntl
    import scripts.cloud.run_vast as launcher
    config = Path("configs/train-8k-local.json").resolve()
    lock_path = tmp_path / ".runtime/experiments/gpu.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(launcher, "REPO", tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_vast", "--gpus", "1", "--config", str(config),
                                         "--data-dir", str(tmp_path / "new")] + setup_flags)
        with pytest.raises(RuntimeError, match="owns the GPU"):
            launch()
    assert not (tmp_path / "new").exists()


def test_legacy_export_checks_transferred_payload_and_leaves_source_unchanged(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    for name in ("model.safetensors","training_state.pt","config.json","run_config.json","tokenizer.json"):
        (source/name).write_bytes(b"fixture:"+name.encode())
    before={p.name:p.read_bytes() for p in source.iterdir()}
    manifest=export_checkpoint_manifest(source,tmp_path/"export.json")
    assert len(verify_checkpoint(source,manifest)) == 64
    assert before == {p.name:p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError,match="unchanged"):
        export_checkpoint_manifest(source,source/"manifest.json")
    (source/"model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_checkpoint(source,manifest)
