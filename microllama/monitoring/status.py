"""Print detached pipeline progress without starting or changing any jobs."""
import json
import os
from pathlib import Path

from ..common.paths import data_root, checkpoint_root, training_log_root


def main():
    root = data_root()
    report = {"data_root": str(root)}
    storage = Path(__file__).resolve().parents[2] / ".runtime/storage.json"
    if storage.exists():
        settings = json.loads(storage.read_text())
        if Path(settings["data_root"]).resolve() == root.resolve():
            report["archive_data_root"] = settings.get("archive_data_root", str(root))
            report["archive_paused"] = settings.get("archive_paused", False)
    status = root / "logs" / "pipeline-status.json"
    if status.exists():
        report.update(json.loads(status.read_text()))
        pid = report["pipeline_pid"]
        try:
            os.kill(pid, 0)
            report["pipeline_running"] = True
        except ProcessLookupError:
            report["pipeline_running"] = False
    raw = root / "raw" / "fineweb-edu-10bt"
    source = None
    complete = []
    if (raw / "manifest.json").exists():
        source = json.loads((raw / "manifest.json").read_text())
        directory = Path(source.get("data_dir", raw))
        complete = [f for f in source["files"] if (directory / f["path"]).exists()
                    and (directory / f["path"]).stat().st_size == f["bytes"]]
        report["dataset"] = {"available_files": len(complete), "total_files": len(source["files"]),
                              "available_bytes": sum(f["bytes"] for f in complete),
                              "total_bytes": source["total_bytes"], "all_verified": source.get("verified", False),
                              "dataset_directory": str(directory), "transfer_complete": source.get("transfer_complete")}
        report["dataset"]["copy_in_progress"] = [{"file": p.name, "buffered_bytes": p.stat().st_size}
                                                     for p in raw.glob("sample/10BT/*.copy-partial")]
    tokens = Path(report.get("token_dir", str(root / "tokens"))) / "manifest.json"
    if tokens.exists():
        preparation = json.loads(tokens.read_text())
        preparation["prepared_chunks"] = len(preparation.pop("shards", []))
        report["preparation"] = preparation
    config_path = Path(report.get("config", "configs/train-8k.json"))
    if config_path.exists():
        config = json.loads(config_path.read_text())
        tracking = Path(".runtime/wandb") / config["run_name"] / "live-run.json"
        if tracking.exists():
            report["wandb"] = json.loads(tracking.read_text())
        report["training_log_root"] = str(training_log_root(root))
        metrics = training_log_root(root) / f"{config['run_name']}.jsonl"
        if metrics.exists() and metrics.stat().st_size:
            with metrics.open("rb") as handle:
                handle.seek(max(0, metrics.stat().st_size - 8192))
                lines = handle.read().decode().splitlines()
            report["training_latest"] = json.loads(lines[-1])
    checkpoints = checkpoint_root(root)
    report["checkpoint_root"] = str(checkpoints)
    locations = [(checkpoints, "checkpoints")]
    if checkpoints.resolve() != (root / "checkpoints").resolve():
        locations.append((root / "checkpoints", "archived_checkpoints"))
    for directory, label in locations:
        for latest in sorted(directory.glob("*/latest.json")):
            report.setdefault(label, {})[latest.parent.name] = json.loads(latest.read_text())
    report["checkpoints_in_progress"] = [{"path": str(path), "buffered_bytes": sum(p.stat().st_size
        for p in path.iterdir() if p.is_file())} for path in checkpoints.glob("*/step-*.partial")]
    archive_status = checkpoints / ".archive-status.json"
    if archive_status.exists():
        report["checkpoint_archive"] = json.loads(archive_status.read_text())
    evaluation = Path(".runtime/evaluation")
    if (evaluation / "launch.json").exists():
        report["evaluation_worker"] = json.loads((evaluation / "launch.json").read_text())
    for path in sorted((evaluation / "results-v2").glob("*/*.json")):
        result = json.loads(path.read_text())
        report.setdefault("quality_reports", []).append({key: result.get(key) for key in
            ("checkpoint", "step", "protocol_version", "text_bits_per_utf8_byte", "cloze_accuracy", "complete", "wandb_url")})
    experiment = Path(__file__).resolve().parents[2] / ".runtime/experiments/current-screen.json"
    if experiment.exists():
        launch = json.loads(experiment.read_text())
        session = Path(launch["session_dir"])
        driver_path = session / "driver-status.json"
        if driver_path.exists():
            driver = json.loads(driver_path.read_text())
            report["experiment_screen"] = driver
            branch = driver.get("active_branch")
            branch_path = session / branch / "status.json" if branch else None
            if branch_path and branch_path.exists():
                state = json.loads(branch_path.read_text())
                report["experiment_branch"] = {key: state.get(key) for key in
                    ("branch", "phase", "pid", "last_step", "valid_targets", "checkpoint", "wandb_url", "error")}
    full_launch = root / "logs/vast-launch.json"
    if full_launch.exists():
        launch = json.loads(full_launch.read_text())
        name = launch["run_name"]
        if Path(name).name == name:
            report["full_training_launch"] = {key: launch.get(key) for key in
                                             ("pid", "run_name", "config", "log", "started_unix")}
            state_path = root / "logs" / name / "status.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                report["full_training"] = state
                try:
                    process = Path(f"/proc/{state['pid']}")
                    command = (process / "cmdline").read_bytes()
                    state["process_alive"] = (b"microllama.training.distributed" in command and
                        str(launch["config"]).encode() in command and
                        (process / "stat").read_text().split()[2] != "Z")
                except (OSError, KeyError) as error:
                    # A sandbox may hide a host process; unavailable is not stopped.
                    state["process_alive"] = None
                    state["process_check_error"] = type(error).__name__
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
