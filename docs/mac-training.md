# Apple Silicon training — validation in progress

**Validation in progress.** This is an experimental MLX continuation of the
published NVIDIA 50B checkpoint. It preserves the Qwen3/Engram model, 8,192-token
context, ordered data cursor, all Adam moments and cumulative valid-target
counters. The ongoing continuation and its long-run convergence are unfinished.
This page describes the implementation and the checks completed so far; it is
not a claim of completed cross-platform accuracy equivalence.

## Implementation and measured configuration

The current training backend is **MLX/Metal**, with native scaled dot-product
attention and fused cross-entropy. All transformer layers execute on the GPU.
The original signed-int64 Engram hashes and deterministic embedding-gradient row
reduction use the CPU. Keeping the reduction deterministic is required by the
exact checkpoint/restart contract. Engram remains enabled.

The selected configuration on the tested M5 Ultra with 96 GiB unified memory is:

| Setting | Value |
| --- | --- |
| Compute | BF16, with framework precision defaults |
| Master weights, Adam moments, accumulated gradients | FP32 |
| Context / effective batch | 8,192 tokens / 16 ordered sequences |
| Physical batch / accumulation | 1 / 16 |
| Loss chunk / allocator cache | 1,024 tokens / 8 GiB |
| Activation recomputation | Disabled for the selected memory budget |
| Projections / attention | Separate projections / stock native SDPA |

Larger batches were measured: batch 2 used about 43.3 GB active memory and was
slightly slower than batch 1 at about 26.9 GB. Earlier batch 4 cases were also slower;
batch 8 was refused when projected memory exceeded the available budget. Larger
unified memory does not by itself imply that a larger physical batch is faster.

The final finite stock screen reached 10,354 valid targets/s. Two full disposable
qualification updates measured 10,018 and 10,065; initial sustained training was
approximately 10.1K. These rates include the ordered global 16 update, normalization,
clipping and Adam, and exclude checkpoint/validation overhead. Historical rates
were about 462 targets/s on M1 Max and about 21K on a single RTX 4090. Those are
different machines/software environments and are not a matched hardware trial.

Optional sorted GPU embedding and custom Metal attention experiments are included
for research. They are **not selected by the default production graph**. Stock
MLX is preferred where custom implementations only offer small gains. This path
uses the GPU rather than the standalone Apple Neural Engine; no matrix-unit
saturation claim is made.

## Completed checks and remaining validation

On the tested working implementation, all 279 parameter derivatives passed an
independent full 8K FP32 CPU reference check (cosine 0.99976037, relative L2
0.02190612, mean-loss difference 0.00002316). The unchanged acceptance limits are
cosine>0.999, relative L2<0.05 and mean-loss difference<0.005.

The stock configuration also passed 128 frozen validation blocks per each of
five sources before and after two disposable updates, and exact next-update
model/Adam/reader/RNG/row replay after save/reload. Production imports the original
Vast checkpoint again and discards tuning/qualification updates. Fresh runs
always need their own source-, option- and hardware-bound qualification;
Private operational receipts are not shipped as reusable approvals.

The old seven-task CPU evaluation at step 383099 / 50,167,011,231 valid targets
completed on the Mac continuation:

| Task (%) | Earlier V2 at 49.237B | Mac V2 at 50.167B |
| --- | ---: | ---: |
| HellaSwag | 43.89 | 43.21 |
| OpenBookQA | 32.40 | 32.80 |
| WinoGrande | 53.43 | 52.88 |
| ARC-Easy | 54.92 | 55.39 |
| ARC-Challenge | 30.12 | 30.63 |
| BoolQ | 60.31 | 61.35 |
| PIQA | 67.46 | 67.30 |
| Seven-task mean | 48.93 | 49.08 |

All full counts, per-example selected scores, finite standard errors and the
arithmetic mean were reconciled. This is the original zero-shot old7 protocol,
CPU FP32, no chat template, no added BOS, prefix token 1, and full examples. It is
separate from the newer official ten-subtask protocol. Different training
exposures and arm64/x86 CPU platforms prevent treating this comparison as an
isolated backend experiment. See [machine-readable results](../results/mac-validation.json)
and [evaluation protocols](evaluation.md).

Long-run convergence, broader capability checks, portability across Apple Silicon
configurations and a complete public end-to-end replay remain **validation in
progress**. Publishing this source did not rerun a new full real-data GPU
qualification or complete the continuation.

## Install and restore inputs

Use Apple Silicon macOS, Python 3.12, `uv`, Git and the `zstd` command available on PATH. Large
inputs, caches and outputs are intentionally restricted to a mounted
`/Volumes/TrainData`; the code refuses an internal-disk substitute. The current
fast launcher's 64 GiB MLX memory cap and 8 GiB cache are intended for the tested
96 GiB system. They are not a supported low-memory preset.

```bash
git clone https://github.com/keeeeenw/MicroLlama2-training.git
cd MicroLlama2-training
# Install uv separately if it is not already available.
bash scripts/mac/setup.sh
uv pip install --python .venv-mac/bin/python -r scripts/mac/requirements-mlx-speed.txt

MAC_ROOT=/Volumes/TrainData/MicroLlamaV2/mac-continuation
MAC_PYTHON=.venv-mac/bin/python
"$MAC_PYTHON" scripts/mac/download_checkpoint.py --root "$MAC_ROOT/checkpoint-download"
```

The downloader verifies the immutable final manifest and every payload against
the canonical public checkpoint repository. Its explicit default payload budget
is 4.7 GB. Use `--max-download-bytes 0` for verified local files only. Dataset/token
archives are a separate input and are not implicitly downloaded by this command.
Authenticate to Hugging Face or W&B under your own account when needed; credentials
belong in normal CLI login/environment storage, never in a published plan.

Before training, restore a verified parent growth catalog, public archive/file
indexes, original validation blocks, and the current mixture chunks from the
[published archives](https://huggingface.co/keeeeenw/MicroLlama2-388M-checkpoints).
`python -m microllama.data_io.mac_staging --help` describes deriving an isolated
catalog with local paths while preserving source inventory hashes and the saved
reader state. Its transfer ceiling defaults to zero. Review the estimated
remaining coverage and choose a finite cumulative target inside it.

## Prepare, qualify, then launch

This is an advanced continuation interface, not a one-command reconstruction of
the full NVIDIA history. First create a data plan with
`python -m microllama.training.mac_run prepare --help`, supplying the immutable
final checkpoint, derived catalog, verified archive/file indexes, original
validation root, archive cache, output and explicit target. The `prepare` command
is an offline proposal; it does not start the legacy PyTorch/MPS trainer.

Once that plan exists as `$MAC_ROOT/data-plan.json`:

```bash
"$MAC_PYTHON" -m microllama.training.mac_ultra_run prepare \
  --data-plan "$MAC_ROOT/data-plan.json" --output "$MAC_ROOT/mlx-plan.json"
"$MAC_PYTHON" scripts/mac/prepare_cpu_oracle.py \
  --plan "$MAC_ROOT/mlx-plan.json" --output "$MAC_ROOT/cpu-reference"
"$MAC_PYTHON" scripts/mac/qualify_ultra_fast.py \
  --plan "$MAC_ROOT/mlx-plan.json" --oracle "$MAC_ROOT/cpu-reference" \
  --output "$MAC_ROOT/qualification" --blocks 128
"$MAC_PYTHON" -m microllama.training.mac_fast_run launch \
  --plan "$MAC_ROOT/mlx-plan.json" --qualification "$MAC_ROOT/qualification" \
  --output "$MAC_ROOT/production"
```

The oracle exports independent CPU gradients and performs no optimizer update.
Qualification is disposable. An 8-block diagnostic is not admitted for production.
The launcher requires the full 128-block receipt and matching current sources,
hardware and options. It freezes code, launches detached under `caffeinate`, uses
a new tracking identity, and starts from the original CUDA checkpoint at
step 381824 / 50,000,054,177 valid targets. Keep framework precision defaults; do
not set `MLX_ENABLE_TF32` for this workflow.

The selected worker saves after its first complete update and every 25 updates,
keeps two complete checkpoints, and validates after 25 updates, every 250 and at
completion. Nonfinite state or a source loss increase above the existing 0.1-nat
guard stops it. A checkpoint, status or setup score is not proof that a full run
has completed. Inspect `run/worker-status.json`, `run/events.jsonl`, tracking status
and `run/checkpoints/latest.json`. A `run/STOP` file or SIGTERM requests a graceful
stop at a complete update; there is no automatic retry or target extension.

For interruption recovery, the same launch command accepts
`--resume-run /Volumes/TrainData/path/to/previous-production/run` with a fresh
output directory. It verifies that run's frozen launcher, checkpoint manifest,
backend options and original-Vast lineage; restores all state; reruns held-out
validation; and continues the same Mac tracking identity. It does not accept an
unrelated old Mac checkpoint as a fresh initial parent. Retention handles macOS
AppleDouble companion files disappearing during deletion while retaining
ownership checks and propagating other errors.

## Tests and code map

CPU fixture tests and native GPU tests are separate. When sharing a machine with
training, select CPU tests and omit GPU-only research modules. A new machine must
run the real-data qualification above before a long continuation.

- `mac_mlx.py`, `mac_mlx_fast.py`: model math and selected native MLX graph.
- `mac_fast_checkpoint.py`, `mac_fast_run.py`: explicit backend identity, complete
  checkpoints, full qualification admission, detached launch and verified resume.
- `mac_staging.py`: archive budgets, source hashes and ordered reader staging.
- `scripts/mac/`: setup, preparation, qualification and finite diagnostics.
- `scripts/evaluation/checkpoint_identity.py`: verify native MLX inputs and load
  actual FP32 weights for the original Torch CPU evaluation.

Legacy PyTorch/MPS and earlier MLX entry points remain for reproducibility and
research. Use the selected fast MLX path described here for this experiment.
