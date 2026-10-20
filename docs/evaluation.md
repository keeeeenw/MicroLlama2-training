# Evaluation and milestone data

The tables in `results/` retain exact saved percentages; displayed values are rounded. These are benchmark results, not the two-example setup checks or new inference executed during release preparation.

## Original seven tasks

The original V1 recipe follows [TinyLlama EVAL.md](https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md). Our V2/reference runs use full native-tokenizer CPU/FP32 evaluation, 8K context, no chat/BOS addition, zero shot and token-normalized `acc_norm` for HellaSwag, ARC-E/C, PIQA and OBQA; WinoGrande/BoolQ use raw `acc`. The evaluated harness version is 0.4.9.1. Full sample counts are 10,042 / 2,376 / 1,172 / 1,838 / 500 / 1,267 / 3,270 respectively.

| Task (%) | V1, reported 50B | V2, 5.762B | V2, 49.237B | Liquid, reported 28T | SmolLM2, reported 4T |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 34.30 | 36.29 | 43.89 | 45.15 | 56.36 |
| ARC-Easy | 39.06 | 51.47 | 54.92 | 68.81 | 68.01 |
| ARC-Challenge | 23.29 | 28.41 | 30.12 | 40.96 | 38.40 |
| PIQA | 64.58 | 64.09 | 67.46 | 67.19 | 71.93 |
| WinoGrande | 51.54 | 51.14 | 53.43 | 55.25 | 58.96 |
| OpenBookQA | 30.60 | 30.20 | 32.40 | 36.20 | 38.20 |
| BoolQ | 53.15 | 58.53 | 60.31 | 58.53 | 61.74 |
| Unweighted mean | **42.36** | **45.73** | **48.93** | **53.16** | **56.23** |

`old7.csv` contains those columns; `old7_trajectory.csv` contains the 45 saved curve points. The 49.237B result is the latest independently reconciled complete suite used in this release. The exact 50.000054B final model completed held-out validation, but no exact-final full benchmark column is asserted. Step 379,000 at 49.630B had an observed automatic suite audit; independent delivery reconciliation did not finish before shutdown, so its scores are not promoted here.

At the late signed-off point, ARC remains substantially behind the references while some other scores are close to Liquid. The 4.22 pp mean gap and ~1/569 token ratio are descriptive; datasets, parameter counts, architectures and tokenizers differ. One repeatedly inspected development trajectory cannot establish a multi-seed trend or causal data-source benefit.

## Expanded SmolLM2-style protocol

`official.csv` contains the separate broader comparison. It uses official SmolLM2-360M base definitions: 2K context, character-normalized likelihood, generated QEM for TriviaQA/GSM8K, five-shot GSM8K, no BoolQ and an unweighted ARC-E/C headline average. Ten subtasks produce nine headline rows.

The latest V2 expanded checkpoint is **38,106,765,829 targets**, mean **32.9008%**, versus our Liquid run's **39.6744%** (gap **6.7736 pp**). These numbers are not the 49.237B seven-task scores. Native V1/V2/Liquid columns are CPU FP32; the SmolLM2 column is a published reference under a different runtime. Liquid uses native BOS and its required different loader. TriviaQA retains all 17,944 rows, including repeated question IDs, rather than deduplicating to fit a mistaken count expectation.

Full protocol pins, uncertainties and the continuation illustrations will be linked with the technical report. The original seven-task exclusion ledger did not cover all expanded tasks; this is not a sealed contamination-free benchmark. This training release includes likelihood/fixed-prompt helpers, not the private server scheduler, recovery observers or historical delivery scripts.
