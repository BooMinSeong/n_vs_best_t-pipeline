#!/bin/bash
#SBATCH --job-name=nvbt_gsm8k_llama
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n79,n80
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Run 3 simulation entrypoints (sim_baselines, equal_mix, v1) for the new
# gsm8kfull_Llama-3.2-3B combo in n_vs_best_t pipeline.

source scripts/slurm_common.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -e

COMBO="gsm8kfull_Llama-3.2-3B"
DIST="outputs/distributions-gsm8kfull_n256_Llama-3.2-3B/distributions.json"

OUT_SIM="outputs/n_vs_best_t/sim_baselines/${COMBO}.csv"
OUT_EQ="outputs/n_vs_best_t/equal_mix/${COMBO}.csv"
OUT_V1="outputs/n_vs_best_t/v1/${COMBO}.csv"

mkdir -p outputs/n_vs_best_t/sim_baselines outputs/n_vs_best_t/equal_mix outputs/n_vs_best_t/v1

echo "=== [1/3] sim_baselines (T=1.0, T=0.1, random_T, consensus_vote) ==="
uv run python scripts/n_vs_best_t/run_sim_baselines_one.py \
    --dist-path "$DIST" --combo-name "$COMBO" --out-csv "$OUT_SIM"

echo "=== [2/3] equal_mix ==="
uv run python scripts/n_vs_best_t/run_equal_mix_one.py \
    --dist-path "$DIST" --combo-name "$COMBO" --out-csv "$OUT_EQ"

echo "=== [3/3] v1 (lb_algorithm_v1 @ B=256) ==="
uv run python scripts/n_vs_best_t/run_v1_one.py \
    --dist-path "$DIST" --combo-name "$COMBO" --out-csv "$OUT_V1"

echo "=== DONE ==="
