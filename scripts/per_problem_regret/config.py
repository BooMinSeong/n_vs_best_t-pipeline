"""Config & shared constants for the per-problem regret analysis.

The substrate is NOT a raw generation pool but the per-problem, per-temperature
*empirical answer distribution* stored in ``distributions.json`` (seeds x n_per_seed
= 6 x 256 = 1536 generations aggregated per (problem, T)).  We therefore estimate
per-problem maj@N accuracy by Monte-Carlo sampling from the multinomial ``P`` matrix
(exactly as ``run_sim_v2_one.py`` does for the dataset-level numbers), NOT by the
bootstrap-from-generations recipe in the spec's section 2.1.  This keeps every
per-problem number on the same currency as the existing dataset-level tables.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Repo layout -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "per_problem_regret"
COMBOS_FILE = REPO_ROOT / "scripts" / "n_vs_best_t" / "combos.txt"

# Strategy / grid definitions (match run_sim_v2_one.py) -----------------------
T_VALUES = [round(0.1 * (i + 1), 1) for i in range(12)]          # 0.1 .. 1.2
SINGLE_T_NAMES = [f"T{t:.1f}" for t in T_VALUES]
MIX_BASELINES = ["random_T", "equal_mix", "consensus_vote"]
ALL_STRATEGIES = SINGLE_T_NAMES + MIX_BASELINES
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]

# Display labels (paper-ready) ------------------------------------------------
DISPLAY_NAME = {
    "random_T": "RandomT",
    "equal_mix": "Temperature Pool",
    "consensus_vote": "Temperature Consensus",
    **{f"T{t:.1f}": f"T={t:.1f}" for t in T_VALUES},
}

# Representative paired comparisons (Qwen3 example in spec section 2.8).
# best_fixed_T / dataset-oracle handled specially in analyze.py.
PAIRED_COMPARISONS = [
    ("T1.0", "equal_mix"),
    ("T1.0", "consensus_vote"),
    ("equal_mix", "consensus_vote"),
    ("T1.0", "best_fixed_T"),
    ("equal_mix", "best_fixed_T"),
]

# Regret thresholds reported as exceedance probabilities.
REGRET_THRESHOLDS = [0.1, 0.2]


@dataclass
class Config:
    """Run configuration; serialised into parquet metadata for reproducibility."""
    n_sims: int = 5000              # B in the spec (per-problem MC replicates)
    seed: int = 42
    n_grid: list[int] = field(default_factory=lambda: list(N_GRID))
    win_tie_threshold: float = 0.01  # |Delta| threshold for win/tie/loss
    dominance_n_grid: int = 200

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_combo(combo: str) -> tuple[str, str]:
    """'aime2025_Qwen3-4B-Instruct-2507' -> ('aime2025', 'Qwen3-4B-Instruct-2507')."""
    dataset, model = combo.split("_", 1)
    return dataset, model


def load_combos() -> dict[str, Path]:
    """Return {combo_name: absolute distributions.json path} from combos.txt."""
    out: dict[str, Path] = {}
    for line in COMBOS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, rel = line.split()
        out[name] = (REPO_ROOT / rel).resolve()
    return out
