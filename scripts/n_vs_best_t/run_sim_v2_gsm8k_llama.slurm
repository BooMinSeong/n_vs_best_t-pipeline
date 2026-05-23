#!/bin/bash
#SBATCH --job-name=nvbt_simv2_gsm8k_llama_v2
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n79,n80
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Generate sim_v2 CSV for gsm8kfull_Llama-3.2-3B so that merge_sim_v2.py can include
# this combo in best_t_table.csv (sim-derived best-T from 12 single-T + 5 mix baselines).

source scripts/slurm_common.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -e

COMBO="gsm8kfull_Llama-3.2-3B"
DIST="outputs/distributions-gsm8kfull_n256_Llama-3.2-3B/distributions.json"
OUT="outputs/n_vs_best_t/sim_v2/${COMBO}.csv"

mkdir -p outputs/n_vs_best_t/sim_v2

uv run python scripts/n_vs_best_t/run_sim_v2_one.py \
    --dist-path "$DIST" --combo-name "$COMBO" --out-csv "$OUT"

echo "=== DONE ==="
