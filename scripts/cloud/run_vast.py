"""One setup/prepare/detached-launch command for single-node CUDA pretraining."""

import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.cloud.hardware import estimate, profiles

REPO = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def export_checkpoint_manifest(source, destination):
    """Hash the source-side restart payload without editing it or copying multi-GB weights."""
    required = {"model.safetensors", "training_state.pt", "config.json", "run_config.json", "tokenizer.json"}
    allowed = required | {"tokenizer_config.json", "source_manifest.json", "special_tokens_map.json",
                          "chat_template.jinja", "generation_config.json", "engram_config.json"}
    files = {}
    for name in sorted(allowed):
        file = source / name
        if not file.is_file():
            continue
        before = file.stat()
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda:handle.read(8 * 2**20), b""):
                digest.update(block)
        after = file.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError(f"Checkpoint payload changed during export: {name}")
        files[name] = {"bytes":after.st_size,"sha256":digest.hexdigest()}
    if not required.issubset(files):
        raise ValueError("Checkpoint export requires a complete restart payload")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Export outside the original checkpoint to keep it unchanged")
    save_json(destination,{"schema_version":1,"source_checkpoint":str(source.resolve()),"files":files,
                           "scope":"Source-side payload identity; legacy checkpoint has no publication receipt"})
    return destination


def default_run_name(args):
    suffix = f"-{args.hardware_profile}" if args.hardware_profile else ""
    return f"fineweb-edu-338m-8k-vast-{'engram' if args.engram else 'dense'}{suffix}"


def configuration(args):
    source = args.resume / "run_config.json" if args.resume else args.config
    config = json.loads(source.read_text())
    config["run_name"] = args.run_name or (config["run_name"] if args.resume else
        default_run_name(args))
    if args.target_tokens is not None:
        config["target_tokens"] = args.target_tokens
    config["eval_batches"] = max(128, config["eval_batches"])
    micro_batch = args.micro_batch_size
    if micro_batch is None and args.hardware_profile and not args.custom_config and not args.resume:
        micro_batch = args.hardware["micro_batch_size"]
    if micro_batch is not None:
        if args.resume and micro_batch != config["micro_batch_size"]:
            raise ValueError("Exact resume preserves batch partitioning; use --continue-from for changes")
        sequences = config["micro_batch_size"] * config["gradient_accumulation_steps"]
        if sequences % micro_batch:
            raise ValueError("Microbatch must divide the existing global sequence batch")
        config["micro_batch_size"] = micro_batch
        config["gradient_accumulation_steps"] = sequences // micro_batch
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["run_name"]):
        raise ValueError("Run name must be a single safe directory name")
    if config["gradient_accumulation_steps"] % args.gpus:
        raise ValueError("GPU count must divide global accumulation")
    return config


def checkpoint_free_bytes(parent=None, keep=2, retained=0):
    """Reserve retained saves plus one publication, with a 32 GiB floor.

    A state-preserving parent supplies the actual full Adam checkpoint size.
    Fresh/weights-only launches retain the conservative 56 GiB allowance because
    adding memory or initializing Adam can enlarge their saved payload.
    Already-published checkpoints in this run occupy retained slots and their
    bytes are already excluded from filesystem free space.
    """
    slots = keep + 1 - min(retained, keep)
    if parent is None:
        return (32 + 8 * slots) * 2**30
    payload = sum((parent / name).stat().st_size for name in
                  ("model.safetensors", "training_state.pt"))
    return 32 * 2**30 + int(payload * slots * 1.1)


def training_command(args, config_path):
    command = [str(args.train_python), "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", f"--nproc-per-node={args.gpus}", "--max-restarts=0", "--module",
               "microllama.training.distributed", "--config", str(config_path), "--token-dir", str(args.token_dir),
               "--checkpoint-dir", str(args.checkpoint_dir), "--checkpoint-keep", str(args.checkpoint_keep)]
    for flag, value in [("--resume", args.resume), ("--continue-from", args.continue_from), ("--init-from", args.init_from),
                        ("--parent-manifest", args.parent_manifest), ("--max-steps", args.max_steps)]:
        if value is not None:
            command += [flag, str(value)]
    for flag, name in (("--growth-catalog", "growth_catalog"), ("--max-new-tokens", "max_new_tokens"),
                       ("--mixture-parent-catalog", "mixture_parent_catalog"),
                       ("--mixture-parent-catalog-sha256", "mixture_parent_catalog_sha256")):
        value = getattr(args, name, None)
        if value is not None:
            command += [flag, str(value)]
    if args.engram:
        command.append("--engram")
    if args.no_wandb:
        command.append("--no-wandb")
    return command


def run(command, env):
    # Credentials remain in environment; do not dump it to logs/status.
    subprocess.run([str(part) for part in command], cwd=REPO, env=env, check=True)


def setup_interpreters(env):
    """Resolve both interpreters before concurrent environment installation."""
    interpreters = {version: shutil.which(f"python{version}") for version in ("3.11", "3.12")}
    if not all(interpreters.values()):
        bootstrap = REPO / ".venv-bootstrap"
        if not (bootstrap / "bin/python").exists():
            run([sys.executable, "-m", "venv", bootstrap], env)
        run([bootstrap / "bin/python", "-m", "pip", "install", "uv>=0.9,<1"], env)
        uv = bootstrap / "bin/uv"
        managed = {**env, "UV_PYTHON_INSTALL_DIR": str(REPO / ".runtime/python")}
        run([uv, "python", "install", "3.11", "3.12"], managed)
        for version in interpreters:
            result = subprocess.check_output([str(uv), "python", "find", version], env=managed, text=True).strip()
            interpreters[version] = result
    return interpreters


def setup_environment(env, interpreters, hardware=None):
    """Install either the data environment or the selected training environment."""
    directory, version, lock = (hardware["environment"], "3.11", hardware["requirements"]) if hardware else (
        ".venv", "3.12", "requirements-data.lock")
    python = REPO / directory / "bin/python"
    if not python.exists():
        run([interpreters[version], "-m", "venv", REPO / directory], env)
    run([python, "-m", "pip", "install", "--upgrade", "pip"], env)
    if hardware:
        run([python, "-m", "pip", "install", f"torch=={hardware['torch_version']}",
             "--index-url", hardware["wheel_index"]], env)
    run([python, "-m", "pip", "install", "-r", REPO / lock], env)
    run([python, "-m", "pip", "install", "--no-deps", "-e", REPO], env)


def setup(env, hardware=None):
    interpreters = setup_interpreters(env)
    setup_environment(env, interpreters)
    setup_environment(env, interpreters, hardware or profiles()["4090-24"])


def prepare(args, env):
    env = {**env, "TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": str(args.threads)}
    if (args.token_dir / "manifest.json").exists():
        summary = json.loads((args.token_dir / "manifest.json").read_text())
        if summary.get("complete") and summary.get("status") != "failed":
            print("Reusing completed tokens; no download or transfer.", flush=True)
            return
    python = REPO / ".venv/bin/python"
    run([python, "-m", "microllama.data_io.tokenizer"], env)
    run([python, "-m", "microllama.data_io.download", "--dry-run"], env)
    logs = args.data_dir / "logs/setup"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for name, command in [("download", [python, "-m", "microllama.data_io.download", "--workers", args.workers]),
                              ("prepare", [python, "-m", "microllama.data_io.prepare", "--output", args.token_dir, "--threads", args.threads])]:
            with (logs / f"{name}.log").open("a") as log:
                child = subprocess.Popen([str(v) for v in command], cwd=REPO, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((name, child))
        while any(child.poll() is None for _, child in children):
            for name, child in children:
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")
            time.sleep(1)
        if any(child.returncode for _, child in children):
            raise RuntimeError("Dataset preparation failed")
    finally:
        for _, child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for _, child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def qualification_cap(args):
    """Bound new updates without deserializing a multi-GB Adam checkpoint."""
    if args.qualification_updates is None:
        return args.max_steps
    parent = args.resume or args.continue_from
    step = 0
    if parent:
        saved = json.loads((parent / "run_config.json").read_text())
        if saved.get("profile") == "engram-confirmation":
            raise ValueError("Promote confirmation first, then qualify from a full-run checkpoint; or specify --max-steps explicitly")
        manifest = parent / "checkpoint_manifest.json"
        if manifest.exists():
            step = json.loads(manifest.read_text())["step"]
        else:
            match = re.fullmatch(r"step-(\d+)", parent.name)
            if not match:
                raise ValueError("Legacy qualification needs a step-NNNNNNNN directory or an explicit --max-steps")
            step = int(match[1])
        if type(step) is not int or step < 0:
            raise ValueError("Invalid checkpoint step")
    return step + args.qualification_updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "TrainData")
    parser.add_argument("--token-dir", type=Path)
    parser.add_argument("--growth-catalog", type=Path, help="Explicit append-only growth catalog; preserves frozen original validation")
    parser.add_argument("--mixture-parent-catalog", type=Path, help="Bound parent catalog for an explicit additive-source continuation")
    parser.add_argument("--mixture-parent-catalog-sha256", help="SHA-256 of the immutable parent catalog snapshot")
    parser.add_argument("--max-new-tokens", type=int, help="Checkpoint and stop after this many additional valid targets")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Hardware profiles default to the current held-rate 8K recipe; legacy calls retain v2")
    parser.add_argument("--gpus", type=int, help="Defaults to four; hardware profiles require four")
    parser.add_argument("--hardware-profile", choices=("4090-24", "4090-48", "5090-32"))
    parser.add_argument("--micro-batch-size", type=int, help="Repartition the existing global batch; e.g. 1 for the 48GB fallback")
    parser.add_argument("--engram", action="store_true", help="Opt-in research architecture; default is dense AdamW")
    parser.add_argument("--target-tokens", type=int, help="Default 10B starting stage; later 50B requires sufficient prepared data")
    parser.add_argument("--run-name")
    parser.add_argument("--setup", action="store_true", help="Create pinned local Python environments")
    parser.add_argument("--prepare", action="store_true", help="Reuse completed tokens or download/tokenize the pinned 10BT sample")
    parser.add_argument("--prepare-only", action="store_true", help="Finish setup/data without starting the GPU trainer")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    bound = parser.add_mutually_exclusive_group()
    bound.add_argument("--max-steps", type=int, help="Optional total-step cap, e.g. 3 for first-machine qualification")
    bound.add_argument("--qualification-updates", type=int, help="Run this many NEW updates, save, then stop; recommend 40 on a new host")
    parser.add_argument("--estimate", action="store_true", help="Print unmeasured 10B/50B runtime and optional cost forecasts; no GPU/filesystem writes")
    parser.add_argument("--hourly-prices", nargs=3, type=float, metavar=("4090_24", "4090_48", "5090_32"),
                        help="With --estimate: whole-instance hourly rates, in profile order")
    parser.add_argument("--measured-tokens-per-second", nargs=3, type=float, metavar=("4090_24", "4090_48", "5090_32"),
                        help="With --estimate: measured effective global rates including validation/save time")
    parser.add_argument("--already-trained-tokens", type=int, default=0, help="With --estimate: subtract parent exposure from the cumulative target")
    parser.add_argument("--single-hourly-price", type=float, help="With --estimate: optional hourly cost assumption for the current owned GPU")
    parser.add_argument("--single-tokens-per-second", type=float, help="With --estimate: measured effective throughput override for the current single GPU")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print config/command without installs, downloads or writes")
    parser.add_argument("--stop", action="store_true", help="Ask rank zero of this run to checkpoint and stop gracefully")
    parser.add_argument("--export-checkpoint", type=Path, help="Hash a local seed checkpoint without changing it")
    parser.add_argument("--export-manifest", type=Path, help="Destination of source-side checksum export")
    parser.add_argument("--parent-manifest", type=Path, help="Uploaded source-side checksum export for a legacy seed")
    transition = parser.add_mutually_exclusive_group()
    transition.add_argument("--resume", type=Path, nargs="?", const=Path("latest"),
                            help="Exact resume; omit path to use this run's latest local publication")
    transition.add_argument("--continue-from", type=Path)
    transition.add_argument("--init-from", type=Path)
    args = parser.parse_args()
    if bool(args.mixture_parent_catalog) != bool(args.mixture_parent_catalog_sha256):
        parser.error("Use both --mixture-parent-catalog and --mixture-parent-catalog-sha256")
    if args.mixture_parent_catalog:
        if not args.growth_catalog or not args.continue_from or not args.engram:
            parser.error("Mixture migration requires --growth-catalog, --continue-from and --engram")
        if args.gpus not in (None,4):
            parser.error("Mixture migration preserves the four saved ranks")
        if not re.fullmatch(r'[0-9a-f]{64}', args.mixture_parent_catalog_sha256):
            parser.error("Mixture parent catalog SHA-256 must be 64 lowercase hexadecimal characters")
        args.mixture_parent_catalog = args.mixture_parent_catalog.expanduser().resolve()
    args.gpus = 4 if args.gpus is None else args.gpus
    args.hardware = profiles()[args.hardware_profile] if args.hardware_profile else None
    if args.hardware and args.gpus != args.hardware["gpus"]:
        parser.error("Hardware profiles require four GPUs")
    if args.estimate:
        if args.setup or args.prepare or args.stop or args.foreground or args.plan:
            parser.error("--estimate cannot launch, prepare, stop or plan training")
        targets = [args.target_tokens] if args.target_tokens is not None else [10_000_000_000, 50_000_000_000]
        print(json.dumps([estimate(target, args.already_trained_tokens, args.hourly_prices,
                                   args.measured_tokens_per_second, args.single_hourly_price,
                                   args.single_tokens_per_second) for target in targets], indent=2))
        return
    if (args.hourly_prices or args.measured_tokens_per_second or args.already_trained_tokens or
            args.single_hourly_price is not None or args.single_tokens_per_second is not None):
        parser.error("Pricing, measured rates and already-trained tokens require --estimate")
    if args.export_checkpoint or args.export_manifest:
        if not args.export_checkpoint or not args.export_manifest or args.plan:
            parser.error("Use --export-checkpoint and --export-manifest together, without --plan")
        print(export_checkpoint_manifest(args.export_checkpoint,args.export_manifest))
        return
    if args.parent_manifest and not (args.resume or args.continue_from or args.init_from):
        parser.error("--parent-manifest requires a checkpoint transition")
    if args.gpus not in (1, 2, 4, 8) or args.workers < 1 or args.threads < 1 or args.checkpoint_keep < 2:
        parser.error("Use 1/2/4/8 GPUs, positive workers/threads and keep at least two checkpoints")
    if args.target_tokens is not None and args.target_tokens < 1:
        parser.error("Token target must be positive")
    if any(value is not None and value < 1 for value in
           (args.micro_batch_size, args.qualification_updates, args.max_steps)):
        parser.error("Microbatch and step/update caps must be positive")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.token_dir = (args.token_dir or args.data_dir / "tokens").expanduser().resolve()
    if args.growth_catalog:
        args.growth_catalog = args.growth_catalog.expanduser().resolve()
        if args.init_from or not (args.resume or args.continue_from):
            parser.error("Growth requires a state-preserving checkpoint transition")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    args.checkpoint_dir = (args.checkpoint_dir or args.data_dir / "checkpoints").expanduser().resolve()
    args.custom_config = args.config is not None
    args.config = (args.config or REPO / ("configs/train-8k-local.json" if args.hardware else
                                        "configs/train-8k-v2.json")).expanduser().resolve()
    # Keep the virtualenv path, not its symlink-resolved system interpreter.
    args.train_python = REPO / (args.hardware["environment"] if args.hardware else ".venv-train") / "bin/python"
    if args.resume == Path("latest"):
        name = args.run_name or default_run_name(args)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("Invalid run name")
        directory = args.checkpoint_dir / name
        latest = json.loads((directory / "latest.json").read_text())
        args.resume = directory / Path(latest["path"]).name
    config = configuration(args)
    args.max_steps = qualification_cap(args)
    config_path = args.data_dir / "logs" / config["run_name"] / "launch-config.json"
    command = training_command(args, config_path)
    if args.plan:
        print(json.dumps({"config": config, "command": command, "data_dir": str(args.data_dir),
                          "hardware_profile": args.hardware_profile, "training_python": str(args.train_python),
                          "global_input_positions_per_update": config["sequence_length"] * config["micro_batch_size"] * config["gradient_accumulation_steps"],
                          "validation_micro_batch_size": 1,
                          "download": "pinned sample/10BT only, standard HF cache; no external transfer",
                          "per_rank_accumulation": config["gradient_accumulation_steps"] // args.gpus}, indent=2))
        return
    if args.stop:
        status = json.loads((config_path.parent / "status.json").read_text())
        if status["phase"] not in ("training", "preflight"):
            raise ValueError("Run is not active")
        pid = status["pid"]
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"microllama.training.distributed" not in cmdline or str(config_path).encode() not in cmdline:
            raise ValueError("Recorded PID no longer identifies this trainer")
        os.kill(pid, signal.SIGTERM)
        print("Stop requested; wait for status=stopped and a published checkpoint.")
        return
    global_lock = None
    if args.setup or not args.prepare_only:
        global_lock_path = REPO / ".runtime/experiments/gpu.lock"
        global_lock_path.parent.mkdir(parents=True, exist_ok=True)
        global_lock = global_lock_path.open("a")
        try:
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A local experiment/training launcher owns the GPU; wait for it to finish") from None
    env = {**os.environ, "MICROLLAMA_DATA_ROOT": str(args.data_dir), "OMP_NUM_THREADS": "2",
           "TOKENIZERS_PARALLELISM": "false"}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.data_dir / ".vast-launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active_path = args.data_dir / "logs/vast-launch.json"
    if active_path.exists():
        prior = json.loads(active_path.read_text())
        prior_pid = prior.get("pid")
        if prior_pid and Path(f"/proc/{prior_pid}/cmdline").exists():
            text = Path(f"/proc/{prior_pid}/cmdline").read_bytes()
            if b"torch.distributed.run" in text:
                raise RuntimeError("A detached training agent is already active in this data root")
    if args.setup:
        setup(env, args.hardware)
    if not args.prepare_only:
        run([args.train_python, "-c", "from scripts.cloud.hardware import cuda_preflight; "
             f"cuda_preflight({args.gpus}, {args.hardware_profile!r})"], env)
        if shutil.which("nvidia-smi"):
            run(["nvidia-smi", "topo", "-m"], env)
    if args.prepare:
        prepare(args, env)
    if args.prepare_only:
        print(f"Setup/data ready at {args.data_dir}; GPU training was not started.")
        return
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    retained = sum(1 for path in (args.checkpoint_dir / config["run_name"]).glob("step-????????")
                   if path.is_dir() and not path.is_symlink() and (path / "checkpoint_manifest.json").is_file())
    required_free = checkpoint_free_bytes(args.resume or args.continue_from, args.checkpoint_keep, retained)
    if shutil.disk_usage(args.checkpoint_dir).free < required_free:
        raise RuntimeError(f"Need {required_free / 2**30:.1f} GiB free for two retained checkpoints, one publication and 32 GiB reserve")
    if not (args.data_dir / "tokenizer/source_manifest.json").exists():
        raise RuntimeError("Tokenizer assets missing; copy them with prepared tokens or use --prepare")
    if (args.checkpoint_dir / config["run_name"] / "latest.json").exists() and not args.resume:
        raise RuntimeError("Run already has checkpoints; use --resume or a distinct run name")
    if (config_path.parent / "metrics.jsonl").exists() and not args.resume:
        raise RuntimeError("Run already has training history; use --resume or a distinct run name")
    save_json(config_path, config)
    if args.foreground:
        run(command, env)
    else:
        log_path = config_path.parent / "train.log"
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock.fileno(), global_lock.fileno()))
        save_json(active_path, {"pid": child.pid, "run_name": config["run_name"], "command": command,
                               "hardware_profile": args.hardware_profile,
                               "config": str(config_path), "log": str(log_path), "started_unix": time.time()})
        print(json.dumps({"detached_pid": child.pid, "log": str(log_path),
                          "status": str(config_path.parent / "status.json")}, indent=2))


if __name__ == "__main__":
    main()
