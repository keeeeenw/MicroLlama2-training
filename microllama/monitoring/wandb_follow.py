"""Publish the running trainer's JSON metrics to W&B without a GPU restart."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ..common.paths import configure_storage, training_log_root
from ..data_io.transfer import atomic_json


def read_records(handle):
    """Read complete appended records; preserve an unfinished trailing line."""
    while True:
        offset = handle.tell()
        line = handle.readline()
        if not line or not line.endswith("\n"):
            handle.seek(offset)
            return
        yield json.loads(line)


def training_metrics(record, target):
    mapping = {"loss": "train/loss", "lr": "train/learning_rate", "grad_norm": "train/gradient_norm",
               "tokens_per_second": "performance/tokens_per_second", "peak_gpu_gib": "performance/peak_gpu_gib",
               "data_epoch": "data/epoch", "corpus_train_tokens": "data/loader_corpus_tokens",
               "validation_loss": "validation/loss"}
    result = {value: record[key] for key, value in mapping.items() if key in record and record[key] is not None}
    result.update({"progress/optimizer_step": record["step"], "progress/training_tokens": record["tokens"],
                   "progress/budget_percent": 100 * record["tokens"] / target})
    return result


def process_alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().split()[2] != "Z"
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-8k.json")
    parser.add_argument("--project", default="MicroLlama2")
    parser.add_argument("--entity")
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--token-dir", type=Path, default=Path(".runtime/tokens"))
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("Poll interval must be positive")
    root = configure_storage()
    config = json.loads(Path(args.config).read_text())
    model = json.loads(Path(config["model_config"]).read_text())
    directory = Path(".runtime/wandb") / config["run_name"]
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "follow.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A W&B publisher for this run is already active")
    import wandb
    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    # This mutation and its access argument are from the pinned SDK's public API.
    project_result = api.client.execute(api.CREATE_PROJECT, {"entityName": entity, "name": args.project, "access": "PRIVATE"})
    if project_result["upsertModel"]["project"]["access"] != "PRIVATE":
        raise RuntimeError("W&B project privacy was not applied")
    fingerprint = hashlib.sha256(json.dumps({"training": config, "model": model}, sort_keys=True).encode()).hexdigest()
    state_path = directory / "live-run.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "id": wandb.util.generate_id(), "project": args.project, "entity": entity,
        "config_sha256": fingerprint, "last_step": 0}
    if (state["config_sha256"], state["project"], state["entity"]) != (fingerprint, args.project, entity):
        raise RuntimeError("W&B run identity changed; use a new training run_name")
    if state["last_step"]:
        remote = api.run(f"{entity}/{args.project}/{state['id']}")
        # Recover records queued locally just before a publisher crash.
        state["last_step"] = min(state["last_step"], remote.summary.get("progress/optimizer_step", 0))
    atomic_json(state_path, state)
    tokenizer = json.loads((root / "tokenizer/source_manifest.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = wandb.init(entity=entity, project=args.project, id=state["id"], resume="allow",
        name=config["run_name"], job_type="pretraining", tags=["from-scratch", "rtx4090", "fineweb-edu", "8k"],
        config={"training": config, "model": model, "tokenizer": tokenizer, "training_code_commit": commit,
                "logging": "JSONL sidecar; training is independent of network availability"},
        dir=str(directory.resolve()), save_code=False,
        settings=wandb.Settings(disable_git=True, disable_code=True, console="off", init_timeout=60))
    state.update(url=run.url, pid=os.getpid(), train_pid=args.train_pid, running=True)
    atomic_json(state_path, state)
    print(f"Live training dashboard: {run.url}", flush=True)
    run.define_metric("progress/optimizer_step")
    for group in ("train/*", "validation/*", "performance/*", "data/*", "progress/*"):
        run.define_metric(group, step_metric="progress/optimizer_step")
    run.define_metric("train/loss", summary="min")
    run.define_metric("validation/loss", summary="min")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    metrics_path = training_log_root(root) / f"{config['run_name']}.jsonl"
    while not metrics_path.exists() and not stopped and process_alive(args.train_pid):
        time.sleep(args.poll_seconds)
    last_snapshot = 0
    try:
        with metrics_path.open() as handle:
            while not stopped:
                for record in read_records(handle):
                    if record["step"] <= state["last_step"]:
                        continue
                    values = training_metrics(record, config["target_tokens"])
                    if time.monotonic() - last_snapshot > 30:
                        manifest = args.token_dir / "manifest.json"
                        if manifest.exists():
                            data = json.loads(manifest.read_text())
                            values.update({"data/prepared_train_tokens": data["train_tokens"],
                                           "data/prepared_validation_tokens": data["validation_tokens"],
                                           "data/prepared_chunks": len(data["shards"]),
                                           "data/preparation_complete": int(data["complete"])})
                        raw = root / "raw/fineweb-edu-10bt"
                        source = json.loads((raw / "manifest.json").read_text())
                        complete = [f for f in source["files"] if (raw / f["path"]).exists()]
                        values["data/copied_source_shards"] = len(complete)
                        values["data/copied_source_bytes"] = sum(f["bytes"] for f in complete)
                        last_snapshot = time.monotonic()
                    run.log(values)
                    state["last_step"] = record["step"]
                    atomic_json(state_path, state)
                if not process_alive(args.train_pid):
                    break
                time.sleep(args.poll_seconds)
        run.summary["trainer_process_running"] = process_alive(args.train_pid)
        run.summary["last_training_step"] = state["last_step"]
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        state["running"] = False
        atomic_json(state_path, state)


if __name__ == "__main__":
    main()
