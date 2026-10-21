#!/usr/bin/env bash
# Small code environment on the Mac; all large caches/data on TrainData.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
from pathlib import Path
if not Path('/Volumes/TrainData').is_mount():
    raise SystemExit('Mount TrainData before setup; refusing an internal SSD substitute.')
PY
export UV_CACHE_DIR=/Volumes/TrainData/MicroLlamaV2/mac-continuation-50b/uv-cache
export HF_HUB_CACHE=/Volumes/TrainData/MicroLlamaV2/mac-continuation-50b/hf-cache
export HF_XET_CACHE=/Volumes/TrainData/MicroLlamaV2/mac-continuation-50b/xet-cache
export UV_LINK_MODE=copy
uv venv --python 3.12.10 .venv-mac
uv pip install --python .venv-mac/bin/python -r scripts/mac/requirements-mac.txt
uv pip install --python .venv-mac/bin/python --no-deps -e .
python3 scripts/mac/clean_appledouble.py .venv-mac
printf '%s\n' 'Environment ready. Dataset staging remains local-only until an explicit byte budget is supplied.'
