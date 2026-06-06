#!/bin/bash -l
# One-time bootstrap for BanditDL on the EPFL Izar GPU cluster.
# Run this from a login node (compute nodes typically have no internet).
#
#   bash slurm/setup.sh           # install uv if missing + uv sync
#   bash slurm/setup.sh --femnist # also pre-download the FEMNIST HF dataset
#
# Safe to re-run; each step is idempotent.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[setup] project root: $PROJECT_ROOT"

# Load native toolchains. uv handles Python itself, so we do not module-load python.
module load gcc 2>/dev/null || echo "[setup] WARN: 'module load gcc' failed; continuing"
module load cuda 2>/dev/null || echo "[setup] WARN: 'module load cuda' failed; continuing"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found; installing the user-local uv from astral.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv under $HOME/.local/bin by default.
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] uv version: $(uv --version)"

echo "[setup] syncing dependencies into .venv (this may take a few minutes the first time)"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-$USER}" uv sync

if [[ "${1:-}" == "--femnist" ]]; then
    HF_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
    echo "[setup] pre-downloading FEMNIST into $HF_CACHE"
    HF_DATASETS_CACHE="$HF_CACHE" uv run python -c "
from banditdl.data.femnist import _load_hf_dataset
dd = _load_hf_dataset()
print('[setup] FEMNIST splits:', {k: len(v) for k, v in dd.items()})
"
fi

echo "[setup] done. Submit jobs with:  sbatch slurm/sbatch_banditdl_gpu.sh <hydra overrides>  (or slurm/sbatch_banditdl_cpu.sh on a CPU cluster)"
