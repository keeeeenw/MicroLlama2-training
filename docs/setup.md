# Setup: Linux and NVIDIA

Use the repository-root quickstart as the single beginner workflow. `scripts.cloud.reproduce` is a small wrapper around `fast_start`; it supports the **first 10B stage only**. It defaults to one RTX 4090 and keeps hardware qualification enabled.

## Preview, then run

```bash
python3 -m scripts.cloud.reproduce --data-dir /mnt/TrainData --plan
python3 -m scripts.cloud.reproduce --data-dir /mnt/TrainData
```

The preview is offline and does not write the data directory. A real invocation installs isolated environments, downloads pinned data, tokenizes it, runs a separate 40-update hardware qualification and then starts a fresh detached production run. Qualification weights, Adam state, RNG and cursors are not reused. A failed qualification stops; there is no blind retry. To use four RTX 5090s, add `--hardware-profile 5090-32 --gpus 4`.

Use `--foreground` to keep the orchestrator attached. `--no-wandb` allows local-only first-stage metrics. Use a fresh `--run-name` for a fresh experiment. The underlying `fast_start --help` exposes advanced options, including explicitly skipping qualification; skipping it is not recommended for a new host.

## Storage and bandwidth

- Choose a real data volume before starting. Environments live inside this checkout; data, token IDs and checkpoints go under `--data-dir`.
- The pinned initial sample is roughly 28.5 GB compressed, followed by roughly 20 GB of token IDs plus boundaries/indexes and checkpoints. Plan around 130 GiB of available workspace; measure actual use.
- The archived complete token corpus is much larger: about 92.6 GB compressed. Do not download all checkpoints or token archives for inference or a short experiment.
- Optional cache override: `export HF_HOME=/mnt/TrainData/hf-cache`. Large caches should live on the data volume.
- Do not use production commands as installation tests on a capped connection. `--plan`, `--help` and CPU tests need no dataset download.

## Manual environments

Use these only when you need direct control; the convenience command installs them automatically. Python 3.12 is used for data tools and Python 3.11 for the pinned CUDA environments. The launcher can install managed interpreters if they are missing. A working host NVIDIA driver is still required.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.lock
.venv/bin/python -m pip install --no-deps -e .

# RTX 4090, CUDA 12.1 wheel stack:
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-train/bin/python -m pip install -r requirements-train.lock
.venv-train/bin/python -m pip install --no-deps -e .

# RTX 5090: use this separate stack instead of the 4090 stack.
python3.11 -m venv .venv-train-blackwell
.venv-train-blackwell/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-train-blackwell/bin/python -m pip install -r requirements-train-blackwell.lock
.venv-train-blackwell/bin/python -m pip install --no-deps -e .
```

Check the available device and BF16 support with the selected environment. A CPU test pass does not qualify your CUDA/NCCL drivers, memory or throughput. Four consumer GPUs can scale poorly over PCIe, so profile before committing to a long rental.

## Track and stop your own run

Status: `DATA_DIR/logs/fast-setup/status.json`; workflow log: `DATA_DIR/logs/fast-setup/workflow.log`. Production metrics/status live in `DATA_DIR/logs/RUN_NAME/` and complete checkpoints in `DATA_DIR/checkpoints/RUN_NAME/`.

Use the low-level launcher's `--stop` with your own data root and run name to request a checkpointed stop. Avoid `kill -9`. `--resume` restores an existing compatible run; `--continue-from` performs a reviewed state-preserving handoff; `--init-from` is weights-only and does not preserve training history. See [training stages](training.md).

Online tracking and checkpoint upload require your own credentials and destinations. No credentials or fixed account/run IDs are supplied. The launcher never rents, reboots, stops or destroys a cloud instance.
