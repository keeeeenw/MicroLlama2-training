"""Run CPU diagnostics on the untrained baseline and each new complete checkpoint."""
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
from microllama.common.paths import data_root, checkpoint_root
from microllama.training.checkpoint_archive import is_alive


def evaluation_complete(output, wandb_run=None, require_native_table=False):
    try:
        report = json.loads(output.read_text())
        return (report.get("complete") is True and (wandb_run is None or (
            report.get("wandb_training_run") == wandb_run and report.get("wandb_status") == "uploaded"
            and (not require_native_table or report.get("wandb_native_table_status") == "published"))))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--run", default="fineweb-edu-338m-8k")
    parser.add_argument("--checkpoint-dir", type=Path, help="Watch a fast local checkpoint root")
    parser.add_argument("--wandb-project", default="MicroLlama2")
    parser.add_argument("--wandb-run", help="Publish fixed prompt Tables to ENTITY/PROJECT/RUN_ID")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--retry-seconds", type=float, default=300,
                        help="Wait this long before retrying a failed or incomplete evaluation")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.retry_seconds <= 0:
        parser.error("Polling and retry intervals must be positive")
    root = data_root()
    checkpoints = args.checkpoint_dir or checkpoint_root(root)
    directory = REPO / ".runtime/evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    retry_after = {}
    initial = root / "checkpoints/initial-338m"
    while True:
        native_required = bool(args.wandb_run) and is_alive(args.train_pid)
        latest = checkpoints / args.run / "latest.json"
        paths = ([initial] if initial.exists() and not args.wandb_run else [])
        if latest.exists():
            paths.append(Path(json.loads(latest.read_text())["path"]))
        for checkpoint in paths:
            output = directory / "results-v2" / checkpoint.parent.name / f"{checkpoint.name}.json"
            if evaluation_complete(output, args.wandb_run, require_native_table=native_required):
                retry_after.pop(checkpoint, None)
                continue
            if time.monotonic() < retry_after.get(checkpoint, 0):
                continue
            print(f"CPU evaluation: {checkpoint}", flush=True)
            command = [sys.executable, "-m", "microllama.evaluation.lightweight",
                       str(checkpoint), "--output", str(output)]
            command.extend(["--wandb-run", args.wandb_run] if args.wandb_run else
                           ["--wandb-project", args.wandb_project])
            if native_required:
                command.extend(["--wandb-train-pid", str(args.train_pid)])
            try:
                result = subprocess.run(command, cwd=REPO, check=False,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                             "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                if result.returncode != 0 or not evaluation_complete(output, args.wandb_run,
                                                                   require_native_table=native_required):
                    retry_after[checkpoint] = time.monotonic() + args.retry_seconds
                    print(f"Evaluation incomplete for {checkpoint}; retrying after {args.retry_seconds:g}s",
                          file=sys.stderr, flush=True)
                else:
                    retry_after.pop(checkpoint, None)
            except OSError as error:
                retry_after[checkpoint] = time.monotonic() + args.retry_seconds
                print(f"Could not launch evaluation for {checkpoint}: {error}; retrying after "
                      f"{args.retry_seconds:g}s", file=sys.stderr, flush=True)
        if not is_alive(args.train_pid) and all(
                evaluation_complete(directory / "results-v2" / checkpoint.parent.name / f"{checkpoint.name}.json",
                                    args.wandb_run)
                for checkpoint in paths):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
