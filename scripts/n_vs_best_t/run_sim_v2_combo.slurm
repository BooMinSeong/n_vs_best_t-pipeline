#!/bin/bash
#SBATCH --job-name=nvbt_simv2
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n79,n80
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Generic sim_v2 launcher. Usage:
#   sbatch --job-name=nvbt_simv2_${COMBO} run_sim_v2_combo.slurm <DATASET> <MODEL>
# DATASET examples: aime2023, gsm8kfull, math1k, math500, mathfull
# Re-runs after winner-vs-other fix in run_sim_v2_one.py.

source scripts/slurm_common.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -e

DATASET="$1"
MODEL="$2"
COMBO="${DATASET}_${MODEL}"
DIST="outputs/distributions-${DATASET}_n256_${MODEL}/distributions.json"
OUT="outputs/n_vs_best_t/sim_v2/${COMBO}.csv"

mkdir -p outputs/n_vs_best_t/sim_v2

if [ ! -f "$DIST" ]; then
    echo "ERROR: $DIST not found"
    exit 1
fi

uv run python scripts/n_vs_best_t/run_sim_v2_one.py \
    --dist-path "$DIST" --combo-name "$COMBO" --out-csv "$OUT"

echo "=== DONE: $COMBO ==="
