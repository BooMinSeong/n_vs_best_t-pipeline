#!/bin/bash
# Common SLURM environment setup. Source this at the top of every job script:
#   source scripts/slurm_common.sh

module purge
module load gnu12/12.2.0
module load cuda/12.6
echo "module load complete"

source ~/.bashrc
source .venv/bin/activate
echo "venv activate complete"

export CC=$(which gcc)
export CXX=$(which g++)
echo "compiler set complete"

# Prolog이 생성한 로컬 SSD 환경변수 로드 (HF_HOME, HF_DATASETS_CACHE, WANDB_DIR 등)
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "sourced prolog env: $ENV_FILE"
    export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_BASE/torchinductor"
    export VLLM_CACHE_ROOT="$LOCAL_JOB_BASE/vllm"
    export TRITON_CACHE_DIR="$LOCAL_JOB_BASE/triton"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    mkdir -p "$VLLM_CACHE_ROOT"
    mkdir -p "$TRITON_CACHE_DIR"
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
    mkdir -p "$HF_DATASETS_CACHE"
    echo "HF_HUB_CACHE restored to $HF_HUB_CACHE"
else
    echo "[WARN] prolog env not found: $ENV_FILE (local SSD caching disabled)"
fi
