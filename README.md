# MicroLlama V2 training — 388M

Thank you for the support for [MicroLlama V1](https://huggingface.co/keeeeenw/MicroLlama). This repository contains the **NVIDIA training and data implementation** for the **387,898,305-parameter Qwen3/Engram MicroLlama V2**.

The NVIDIA run completed **50,000,054,177 valid next-token targets** at step **381,824**. My original compute goal was $500; **I spent $880 on compute**, excluding storage, electricity and personal hardware. This is an author-reported expense, not an audited all-inclusive cost.

- **Run inference:** [MicroLlama2-base](https://huggingface.co/keeeeenw/MicroLlama2-base).
- **Full checkpoints and token archives:** [MicroLlama2-388M-checkpoints](https://huggingface.co/keeeeenw/MicroLlama2-388M-checkpoints).
- **Technical report:** link coming shortly.
- **Training code:** this repository. The clean release history starts here; operational logs and private infrastructure records are excluded.

This is **not** the earlier ~500M Llama-3.2/SlimPajama experiment in [MicroLlama2-checkpoints](https://huggingface.co/keeeeenw/MicroLlama2-checkpoints).

## Completed milestones

| Milestone | Valid targets | Status / seven-task mean |
| --- | ---: | --- |
| Early V2 checkpoint | 5,761,749,787 | **45.73%**, versus V1's reported-50B **42.36%**: **+3.37 pp** |
| Initial FineWeb-Edu stage | 10,000,022,979 | Complete; **45.90%** |
| End of two-source continuation | 23,773,658,973 | Complete; **47.76%** |
| First four-source comparison | 25,928,550,558 | Complete; **49.00%** |
| Late independently reconciled benchmark | 49,237,409,208 | Seven full tasks complete; **48.93%** |
| NVIDIA endpoint | 50,000,054,177 | Training and frozen validation complete; exact-final full benchmark **not claimed** |

At 49.237B targets the mean is **4.22 pp behind Liquid LFM2.5-350M-Base**, at about **0.176% of its reported 28T tokens**. Architectures, model sizes, tokenizers and data differ: this is a descriptive ratio, not a controlled training-efficiency claim. The expanded SmolLM2-style evaluation is a separate protocol; its latest V2 column is **38.107B**, not the final checkpoint. See [evaluation and tables](docs/evaluation.md).

## Start with one workflow

The supported convenience command below starts a **fresh first-10B experiment**, with pinned source hashes, parallel preparation and an isolated hardware qualification. It does not resume the published final model or recreate the entire historical 50B trajectory in one command.

Requirements: Linux, Python 3.11+, Git, a working NVIDIA driver, one RTX 4090 24GB or four RTX 5090 32GB GPUs. Setup installs the selected Python environments and CUDA-compatible PyTorch wheels. Allow roughly 130 GiB for the first stage, on a dedicated data volume. Do not use a metered connection without reviewing transfer sizes.

```bash
git clone https://github.com/keeeeenw/MicroLlama2-training.git
cd MicroLlama2-training

# Preview only: no downloads, installations or GPU training.
python3 -m scripts.cloud.reproduce --data-dir /mnt/TrainData --plan

# Fresh 10B run on one RTX 4090; detached by default.
python3 -m scripts.cloud.reproduce --data-dir /mnt/TrainData

# Alternative: four RTX 5090s.
python3 -m scripts.cloud.reproduce --data-dir /mnt/TrainData \
  --hardware-profile 5090-32 --gpus 4
```

Authenticate to W&B under **your own account**, or pass `--no-wandb` for the first-stage workflow. Optional Hugging Face login can improve download limits. Never put tokens in source files or command-line arguments. See [setup](docs/setup.md) for storage, manual installation, stopping and resume.

## The actual 50B data recipe

1. FineWeb-Edu through **10.000B** valid targets.
2. **60% FineWeb-Edu / 40% DCLM-Edu** through **23.774B**.
3. **48% FineWeb-Edu / 32% DCLM-Edu / 15% Cosmopedia v2 / 5% FineMath 4+** for subsequent targets through **50B**.

Cosmopedia is an existing public synthetic corpus. Source weights apply to future targets; never rebase old exposure counts or restart source cursors at a transition. See [stages and checkpoint restoration](docs/training.md). The original two-source selection remains available as historical data metadata; it is not the final four-source recipe.

## Scope and reproducibility

The model uses 24 layers, width 1,024, an 8K context, a 337.695M Qwen3-style backbone and 50.204M Engram-style memory parameters. NVIDIA training uses BF16 compute with FP32 parameters/gradients/Adam state, boundary-masked loss and a constant learning rate after warmup. Engram is a lightweight prototype, not the complete architecture from its inspiration paper.

A new first-stage run uses the selected architecture from initialization. The published trajectory included an early dense-to-Engram development handoff, hardware changes and later source transitions; the convenience command is not a bitwise replay of that history. Complete public restart checkpoints preserve the state needed for continuation. Exact restoration requires compatible archived token inventories, world size and directory layout.

Training deliberately pins **Transformers 4.57.6**. Inference in `MicroLlama2-base` uses its separately tested modern adapter. Do not upgrade the training environment to Transformers 5 merely to match inference.

CPU tests and dry-run entry points are included. A full new CUDA qualification or complete 50B replay was **not** run during release preparation. Mac/MLX continuation, private recovery capsules, provider control and operational monitoring histories are outside this release.

## Code map

- `microllama/training/`: model, masked objective, distributed training and restart validation.
- `microllama/data_io/`: pinned source preparation, document boundaries, deduplication and mixture readers.
- `scripts/cloud/reproduce.py`: focused first-stage entry point.
- `scripts/cloud/run_vast.py`: low-level Linux/NVIDIA launcher; it does not rent an instance.
- `scripts/cloud/grow.py`: historical two-source gate implementation, not the completed four-source workflow.
- `configs/`: model/settings and source inventories; inspect stage and historical status before reuse.
- `results/`: published-score tables, without private operational receipts.
- `tests/`: CPU/fixture tests; hardware-specific checks require your machine.

Apache 2.0, with retained third-party notices. See [attribution](THIRD_PARTY_NOTICES.md) and [publication/privacy scope](SECURITY.md).
