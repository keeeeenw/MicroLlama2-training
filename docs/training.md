# Training stages and full checkpoints

## What the convenience command reproduces

`python3 -m scripts.cloud.reproduce` creates a fresh **first-10B** experiment with the selected 8K architecture. It does not automatically recreate every historical development/transition step. The previous two-source-to-50B wrapper has deliberately not been presented as the completed run: that run adopted four sources at 23.774B.

The training implementation supports saved-state continuation, two-source/four-source mixture readers and strict validation of optimizer, RNG, source counters and policy. The public checkpoint/data archive supplies immutable artifacts; this release does not package a new unattended, one-command replay of the complete historical 50B run.

## Actual completed schedule

| Stage | Endpoint | Future valid-target mixture |
| --- | ---: | --- |
| Initial | 10,000,022,979 | FineWeb-Edu |
| Two source | 23,773,658,973 | 60% FineWeb-Edu, 40% DCLM-Edu (educational score >=3) |
| Four source | 50,000,054,177 | 48% FineWeb-Edu, 32% DCLM-Edu, 15% broad Cosmopedia v2, 5% FineMath 4+ |

All stages use 8,192-token rows and 131,072 global input positions/update. Valid-target accounting excludes padding and cross-document targets and includes document EOS. The last update may overshoot a finite target by less than 131,072 valid targets. Do not substitute nominal input positions for the saved valid-target counter.

At the four-source transition, the parent exposure was 23,773,658,973: FineWeb-Edu 8,264,178,373 and DCLM-Edu 5,509,457,621 after the original 10,000,022,979-target stage. New 48/32/15/5 weights apply **after** this origin. Do not rebalance past data, restart source cursors, replay consumed chunks or use a new catalog simply because its filenames look similar.

The original dataset inventories remain in `configs/data-recipes/`; `four-source-selected-shards.json` and the specialist source manifests describe added sources. `configs/four-source-continuation.json` is the historical transition-preparation plan, not a current request to acquire data or pause a run. `scripts/cloud/grow.py` implements the older two-source gate only.

## Choose the right checkpoint

- For generation, use [MicroLlama2-base](https://huggingface.co/keeeeenw/MicroLlama2-base), which downloads inference assets only.
- For training/research, use [MicroLlama2-388M-checkpoints](https://huggingface.co/keeeeenw/MicroLlama2-388M-checkpoints).
- The final payload is [`step-00381824`](https://huggingface.co/keeeeenw/MicroLlama2-388M-checkpoints/tree/527ab4101332284704765e1b2f3cfb577609bf56/checkpoint/fineweb-edu-338m-8k-local-engram-4090/step-00381824): 11 files, 4,661,856,785 bytes.
- Its manifest SHA-256 is `9968e0d8db74a6618750ca38c71e68332c3947921f8dd75a591415afd965840c`.
- Token data is pinned at `af9e5b1fa4455991d46aa2c4a623b905942b8053`. It includes 138 archives / 6,800 chunks; download selected archives only.

A narrowly scoped download, requiring Hugging Face Hub's CLI:

```bash
hf download keeeeenw/MicroLlama2-388M-checkpoints \
  --revision 527ab4101332284704765e1b2f3cfb577609bf56 \
  --include 'checkpoint/fineweb-edu-338m-8k-local-engram-4090/step-00381824/*' \
  --local-dir /mnt/TrainData/final-checkpoint
```

This downloads the full 4.66 GB restart payload; it is not an offline preview. Review size and storage first. A safetensors weight file alone cannot restore Adam, all-rank RNG and data order.

## Resume requirements

1. Verify the manifest and each payload hash before loading a checkpoint.
2. Restore tokenizer, exact compatible catalogs, chunk bytes, document-boundary sidecars and frozen validation. Follow the archive's `reproduction/four-source-v1/` layout metadata. Paths embedded in old inventories are part of their identity; do not edit their bytes in place.
3. Preserve the saved architecture marker `engram_qwen3`, normalized identity map and four lookup heads. A stock Qwen3 loader cannot replace the Engram implementation.
4. Preserve all optimizer moments, distributed world-size/RNG state, policy origins and source cursors. Use `--resume` for the same run or a reviewed `--continue-from` handoff; never substitute `--init-from` for exact continuation.
5. Use the pinned training environment and qualify your hardware. The completed distributed endpoint has four-rank state. Do not claim exact four-to-one-rank resume by merely changing `--gpus`.
6. A final 50B checkpoint already reached its finite target. Continuing it requires a new explicit training budget/configuration. It must not silently extend its endpoint.

Inspect `python -m scripts.cloud.run_vast --help` for the generic prepared-data, catalog and continuation flags. The public release is intended to make the actual implementation inspectable and reusable; it does not claim a new complete archive-restore/GPU replay was performed during publication preparation.
