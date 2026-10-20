# Contributing

Use a separate branch and run the CPU suite in a pinned training environment:

```bash
python -m pip install --no-deps -e .
python -m pytest -q
python -m scripts.cloud.reproduce --data-dir /tmp/microllama-plan-only --plan
```

The dry run must not install packages, acquire data or start training. GPU profiling/qualification is separate and must run on the intended hardware. Avoid adding large data, checkpoints, logs, authentication files or private operational records. Run `gitleaks git --redact` before pushing; the supplied config only exempts one known-public tokenizer digest.

Keep the inference adapter's modern dependencies separate from the pinned NVIDIA training code. Preserve checkpoint/source hashes, mixture origins, source cursors, loss masking and frozen validation identities when making compatibility changes. A passing fixture is not a new model benchmark.
