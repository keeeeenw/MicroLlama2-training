"""Overlap verified downloads, chunk preparation, CUDA training and archival."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from microllama.common.paths import configure_storage, checkpoint_root
from microllama.data_io.transfer import atomic_json


def validate_prepared_data(token_dir, tokenizer_manifest):
    summary = json.loads((token_dir / "manifest.json").read_text())
    tokenizer = json.loads(tokenizer_manifest.read_text())
    if summary.get("complete") is not True or summary.get("status") == "failed":
        raise RuntimeError("--prepared-only requires a complete token corpus")
    if summary.get("tokenizer_sha256") != tokenizer["tokenizer_sha256"]:
        raise RuntimeError("Prepared data tokenizer mismatch")
    if not summary.get("train_tokens"):
        raise RuntimeError("Prepared corpus has no training tokens")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-8k.json")
    parser.add_argument("--dataset-location", choices=["cache", "external"], default="cache")
    parser.add_argument("--token-dir", type=Path, help="Override prepared tokens; external workflow stages on NVMe")
    parser.add_argument("--checkpoint-dir", type=Path, help="Fast local saves, asynchronously archived to the data root")
    parser.add_argument("--wandb-project", default="MicroLlama2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Pause after this total optimizer step without changing the token schedule")
    parser.add_argument("--prepared-only", action="store_true",
                        help="Train from a complete local token corpus; skip download, preparation and token archival")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    os.chdir(REPO)
    root = configure_storage()
    args.checkpoint_dir = (args.checkpoint_dir or checkpoint_root(root)).resolve()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.token_dir = args.token_dir or (root / "tokens" if args.dataset_location == "cache" else REPO / ".runtime/tokens")
    lock = (root / "cache/pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another pipeline is already running")
    data_python = str(REPO / ".venv/bin/python")
    train_python = str(REPO / ".venv-train/bin/python")
    local = REPO / ".runtime"
    local.mkdir(exist_ok=True)
    jobs = {}
    status = {"pipeline_pid": os.getpid(), "started_unix": time.time(), "config": args.config,
              "token_dir": str(args.token_dir), "checkpoint_root": str(args.checkpoint_dir), "mode": "prepared" if args.prepared_only else "streaming", "max_steps": args.max_steps}

    def report(phase, **extra):
        status.update(phase=phase, updated_unix=time.time(), jobs={name: {
            "pid": job.pid, "running": job.poll() is None}
            for name, job in jobs.items()}, **extra)
        atomic_json(root / "logs/pipeline-status.json", status)

    def launch(name, command):
        log = local / f"{name}.log"
        with log.open("a") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                     cwd=REPO, stdin=subprocess.DEVNULL)
        jobs[name] = child
        print(f"Started {name} PID {child.pid}; log {log}", flush=True)
        return child

    def running(name):
        job = jobs[name]
        return job.poll() is None

    def succeeded(name):
        job = jobs[name]
        return job.returncode == 0

    raw_manifest = root / "raw/fineweb-edu-10bt/manifest.json"
    try:
        if args.prepared_only:
            validate_prepared_data(args.token_dir, root / "tokenizer/source_manifest.json")
        if not raw_manifest.exists() and not args.prepared_only:
            subprocess.run([data_python, "-m", "microllama.data_io.download", "--dry-run"], check=True)
        if not (root / "tokenizer/source_manifest.json").exists():
            subprocess.run([data_python, "-m", "microllama.data_io.tokenizer"], check=True)
        if args.prepared_only:
            pass
        elif (not json.loads(raw_manifest.read_text()).get("verified") or
              (args.dataset_location == "external" and
               Path(json.loads(raw_manifest.read_text()).get("data_dir", raw_manifest.parent)).resolve() != raw_manifest.parent.resolve())):
            command = [data_python, "-m", "microllama.data_io.download"]
            if args.dataset_location == "external":
                command += ["--output", str(raw_manifest.parent)]
            launch("download", command)
        prepared_manifest = args.token_dir / "manifest.json"
        prepared_complete = prepared_manifest.exists() and json.loads(prepared_manifest.read_text()).get("complete")
        if args.prepared_only:
            pass
        elif prepared_complete:
            validate_prepared_data(args.token_dir, root / "tokenizer/source_manifest.json")
        else:
            launch("prepare-stream", [data_python, "-m", "microllama.data_io.prepare", "--output", str(args.token_dir)])
        config = json.loads((REPO / args.config).read_text())
        trainer = None
        while True:
            summary_path = args.token_dir / "manifest.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if summary.get("status") == "failed":
                raise RuntimeError(f"Token preparation failed: {summary.get('error')}")
            if "prepare-stream" in jobs and not running("prepare-stream") and (not succeeded("prepare-stream") or not summary.get("complete")):
                raise RuntimeError("Token producer exited before completing the corpus")
            raw_state = json.loads(raw_manifest.read_text()) if raw_manifest.exists() else {}
            raw_complete = raw_state.get("transfer_complete", raw_state.get("verified", False)) and raw_state.get("verified", False)
            if "download" in jobs and not running("download") and not raw_complete:
                raise RuntimeError("Raw-file copying exited before all destination checksums passed")
            if trainer is None and summary.get("train_tokens", 0) > config["sequence_length"]:
                validation_ready = any(json.loads(p.read_text())["splits"]["validation"]["tokens"] > config["sequence_length"]
                    for p in args.token_dir.glob("*/manifest.json") if not p.parent.name.endswith(".partial"))
                if validation_ready:
                    command = [train_python, "-m", "microllama.training.cuda", "--config", args.config,
                               "--streaming-data", "--token-dir", str(args.token_dir),
                               "--checkpoint-dir", str(args.checkpoint_dir)]
                    if args.max_steps is not None:
                        command += ["--max-steps", str(args.max_steps)]
                    latest = args.checkpoint_dir / config["run_name"] / "latest.json"
                    if not latest.exists():
                        latest = root / "checkpoints" / config["run_name"] / "latest.json"
                    initial = root / "checkpoints/initial-338m"
                    if latest.exists():
                        command += ["--resume", json.loads(latest.read_text())["path"]]
                    elif initial.exists():
                        command += ["--init-from", str(initial)]
                    trainer = launch("train", command)
                    if not args.no_wandb:
                        tracking = [train_python, "-m", "microllama.monitoring.wandb_follow", "--config", args.config,
                                    "--train-pid", str(trainer.pid), "--token-dir", str(args.token_dir),
                                    "--project", args.wandb_project]
                        if args.wandb_entity:
                            tracking += ["--entity", args.wandb_entity]
                        launch("wandb", tracking)
            if (trainer is not None and args.checkpoint_dir.resolve() != (root / "checkpoints").resolve()
                    and "checkpoint-archive" not in jobs):
                pid = trainer.pid
                launch("checkpoint-archive", [data_python, str(REPO / "scripts/maintenance/archive_checkpoints.py"),
                    "--source", str(args.checkpoint_dir), "--destination", str(root / "checkpoints"),
                    "--train-pid", str(pid)])
            needs_archive = not args.prepared_only and args.token_dir.resolve() != (root / "tokens").resolve()
            if raw_complete and needs_archive and "archive" not in jobs:
                launch("archive", [data_python, "-m", "microllama.data_io.archive_tokens", "--source", str(args.token_dir)])
            if "archive" in jobs and not running("archive") and not succeeded("archive"):
                raise RuntimeError("Token archive failed; the local cache is retained")
            if trainer is not None and not running("train") and not succeeded("train"):
                raise RuntimeError(f"Training exited without a successful final result; see {local / 'train.log'}")
            archive_complete = (not needs_archive and summary.get("complete")) or ("archive" in jobs and not running("archive"))
            checkpoints_archived = "checkpoint-archive" not in jobs or (
                not running("checkpoint-archive") and succeeded("checkpoint-archive"))
            if trainer is not None and not running("train") and archive_complete and checkpoints_archived:
                report("finished", child_pid=None)
                return
            phase = "preparing_stream" if trainer is None else (
                "training_with_copy" if running("train") and not raw_complete and not args.prepared_only else
                "training" if running("train") else "training_finished_archiving")
            report(phase, child_pid=trainer.pid
                       if trainer is not None and running("train") else None,
                   phase_log=str(local / "train.log") if trainer else str(local / "prepare-stream.log"),
                   prepared_train_tokens=summary.get("train_tokens", 0), raw_complete=raw_complete,
                   checkpoint_archive_error="Checkpoint archiver exited; local saves remain available. See .runtime/checkpoint-archive.log"
                       if "checkpoint-archive" in jobs and not running("checkpoint-archive") and not succeeded("checkpoint-archive") else None,
                   tracking_error="W&B publisher exited; GPU training continues. See .runtime/wandb.log"
                       if "wandb" in jobs and not running("wandb") and not succeeded("wandb") else None)
            time.sleep(10)
    except Exception as error:
        report("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
