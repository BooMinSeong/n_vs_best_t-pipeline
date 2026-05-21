"""Parse outputs/*_n256_*-difficulty/difficulty_temperature_report.md into long-format CSV.

Source filter: only `_n256_` dirs that contain N=256 row in their Maj Method tables.
AIME 4 years (2023-2026) are merged into a single `dataset_family=aime` via
problem-count weighted mean per (model, level, N, T) cell.
Overall stratum is computed as problem-count weighted mean over L1..L5.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs")
DEFAULT_OUT = OUTPUTS_DIR / "n_vs_best_t" / "long"

EXPECTED_TEMPS = [round(0.1 * i, 1) for i in range(1, 13)]  # 0.1..1.2
EXPECTED_N = [1, 2, 4, 8, 16, 32, 64, 128, 256]

DIR_RE = re.compile(
    r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
    r"_n256_"
    r"(?P<model>.+)"
    r"-difficulty$"
)

LEVEL_HEADER_RE = re.compile(r"^## Level (\d+):\s*([\d.]+)-([\d.]+) Accuracy")
MAJ_SECTION_RE = re.compile(r"^### Maj Method\s*$")
PROBLEM_COUNT_RE = re.compile(r"^\*\*(\d+) problems\*\*")
CELL_RE = re.compile(r"^([0-9.]+)\s*±\s*([0-9.]+)$")


@dataclass
class LevelTable:
    level: int
    n_problems: int
    # data[N][T] = (mean, std); only filled when value != "-"
    data: dict[int, dict[float, tuple[float, float]]]


def parse_report(md_path: Path) -> tuple[dict[str, str], list[LevelTable]]:
    """Return (header_info, list_of_LevelTable).

    header_info has: model, dataset (the raw dataset name from dir like 'aime2023').
    """
    m = DIR_RE.match(md_path.parent.name)
    if not m:
        raise ValueError(f"Cannot parse dir name: {md_path.parent.name}")
    info = {"dataset": m.group("dataset"), "model": m.group("model")}

    text = md_path.read_text()
    lines = text.splitlines()

    tables: list[LevelTable] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        lvl_match = LEVEL_HEADER_RE.match(line)
        if not lvl_match:
            i += 1
            continue

        level = int(lvl_match.group(1))
        # find problem count
        n_problems = -1
        j = i + 1
        while j < min(i + 8, len(lines)):
            pc_match = PROBLEM_COUNT_RE.match(lines[j].strip())
            if pc_match:
                n_problems = int(pc_match.group(1))
                break
            j += 1
        # find ### Maj Method
        while j < len(lines) and not MAJ_SECTION_RE.match(lines[j]):
            # bail if we hit next ## Level (no Maj section)
            if LEVEL_HEADER_RE.match(lines[j]):
                break
            j += 1
        if j >= len(lines) or not MAJ_SECTION_RE.match(lines[j]):
            i = j
            continue

        # Maj table header is 2 lines down (header row, separator row), then data
        # Find header row starting with '| N |'
        while j < len(lines) and not lines[j].lstrip().startswith("| N |"):
            j += 1
        if j >= len(lines):
            i = j
            continue
        header_cols = [c.strip() for c in lines[j].strip().strip("|").split("|")]
        temp_cols = []
        for c in header_cols[1:]:
            # 'T0.1' → 0.1
            assert c.startswith("T"), f"Bad temp col: {c}"
            temp_cols.append(round(float(c[1:]), 1))
        assert temp_cols == EXPECTED_TEMPS, (
            f"T grid mismatch in {md_path}: got {temp_cols}, expected {EXPECTED_TEMPS}"
        )
        j += 2  # skip separator row

        data: dict[int, dict[float, tuple[float, float]]] = {}
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            row_cols = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            if not row_cols[0].isdigit():
                j += 1
                continue
            N = int(row_cols[0])
            cells = row_cols[1:]
            row_data: dict[float, tuple[float, float]] = {}
            for T, cell in zip(temp_cols, cells):
                if cell == "-":
                    continue
                cm = CELL_RE.match(cell)
                if cm:
                    row_data[T] = (float(cm.group(1)), float(cm.group(2)))
            if row_data:
                data[N] = row_data
            j += 1

        tables.append(LevelTable(level=level, n_problems=n_problems, data=data))
        i = j

    return info, tables


def tables_to_long_rows(info: dict[str, str], tables: list[LevelTable]) -> list[dict]:
    """Convert per-level tables (per dir) into long rows (one per (level, N, T))."""
    rows = []
    for tbl in tables:
        for N, row in tbl.data.items():
            for T, (mean, std) in row.items():
                rows.append({
                    "model": info["model"],
                    "dataset": info["dataset"],
                    "stratum": f"L{tbl.level}",
                    "level_n_problems": tbl.n_problems,
                    "N": N,
                    "T": T,
                    "mean_acc": mean,
                    "std_acc": std,
                    "source": "markdown",
                })
    return rows


def compute_overall_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset, N, T), compute problem-count weighted mean over L1..L5."""
    overall_rows = []
    grp_cols = ["model", "dataset", "N", "T"]
    for keys, sub in df.groupby(grp_cols):
        # Each level appears once (after AIME merge or single-year)
        weights = sub["level_n_problems"].to_numpy(dtype=float)
        means = sub["mean_acc"].to_numpy(dtype=float)
        stds = sub["std_acc"].to_numpy(dtype=float)
        total_w = weights.sum()
        if total_w <= 0:
            continue
        mean_o = (weights * means).sum() / total_w
        # pooled within-group std (treat each level as independent seed-noise group)
        std_o = float(np.sqrt((weights * stds**2).sum() / total_w))
        overall_rows.append({
            "model": keys[0],
            "dataset": keys[1],
            "stratum": "overall",
            "level_n_problems": int(total_w),
            "N": keys[2],
            "T": keys[3],
            "mean_acc": float(mean_o),
            "std_acc": std_o,
            "source": "markdown",
        })
    return pd.DataFrame(overall_rows)


def merge_aime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine aime2023..aime2026 rows per (model, stratum, N, T) via problem-count weighted mean."""
    is_aime = df["dataset"].str.startswith("aime")
    aime = df[is_aime].copy()
    non_aime = df[~is_aime].copy()

    if aime.empty:
        non_aime["dataset_family"] = non_aime["dataset"]
        return non_aime

    grp_cols = ["model", "stratum", "N", "T"]
    rows = []
    for keys, sub in aime.groupby(grp_cols):
        weights = sub["level_n_problems"].to_numpy(dtype=float)
        means = sub["mean_acc"].to_numpy(dtype=float)
        stds = sub["std_acc"].to_numpy(dtype=float)
        total_w = weights.sum()
        if total_w <= 0:
            continue
        mean_a = float((weights * means).sum() / total_w)
        std_a = float(np.sqrt((weights * stds**2).sum() / total_w))
        rows.append({
            "model": keys[0],
            "dataset": "aime",
            "stratum": keys[1],
            "level_n_problems": int(total_w),
            "N": keys[2],
            "T": keys[3],
            "mean_acc": mean_a,
            "std_acc": std_a,
            "source": "markdown",
        })
    aime_merged = pd.DataFrame(rows)
    aime_merged["dataset_family"] = "aime"
    non_aime["dataset_family"] = non_aime["dataset"]
    return pd.concat([non_aime, aime_merged], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--require-n256", action="store_true", default=True,
                    help="Drop dirs whose Maj table lacks N=256 (default on)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dirs = sorted([
        d for d in args.outputs_dir.iterdir()
        if d.is_dir() and DIR_RE.match(d.name)
        and (d / "difficulty_temperature_report.md").exists()
    ])
    print(f"Found {len(candidate_dirs)} candidate _n256_ difficulty dirs")

    all_rows = []
    dropped = []
    for d in candidate_dirs:
        md = d / "difficulty_temperature_report.md"
        try:
            info, tables = parse_report(md)
        except Exception as e:
            print(f"  SKIP {d.name}: parse error {e}")
            dropped.append((d.name, f"parse:{e}"))
            continue

        # require N=256 in at least one level
        has_n256 = any(256 in tbl.data for tbl in tables)
        if args.require_n256 and not has_n256:
            print(f"  DROP {d.name}: N=256 missing")
            dropped.append((d.name, "no_n256"))
            continue

        rows = tables_to_long_rows(info, tables)
        if not rows:
            print(f"  DROP {d.name}: empty maj data")
            dropped.append((d.name, "empty"))
            continue
        all_rows.extend(rows)
        print(f"  OK {d.name}: {len(rows)} rows")

    if not all_rows:
        raise SystemExit("No data parsed.")

    df = pd.DataFrame(all_rows)

    # Save raw per-year before AIME merge
    raw_path = args.out_dir / "raw_per_year.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote raw per-year: {raw_path} ({len(df)} rows)")

    # Merge AIME
    df_merged = merge_aime(df)
    print(f"After AIME merge: {len(df_merged)} rows, {df_merged.groupby(['model','dataset']).ngroups} (model,dataset) combos")

    # Compute overall stratum
    overall = compute_overall_stratum(df_merged)
    overall["dataset_family"] = overall["dataset"]
    print(f"Overall stratum: {len(overall)} rows")

    long_all = pd.concat([df_merged, overall], ignore_index=True)
    long_all = long_all.sort_values(["model", "dataset", "stratum", "N", "T"]).reset_index(drop=True)

    long_path = args.out_dir / "long_all.csv"
    long_all.to_csv(long_path, index=False)
    print(f"Wrote: {long_path} ({len(long_all)} rows)")

    # Summary
    combos = long_all.groupby(["model", "dataset"]).size().reset_index().rename(columns={0: "n_rows"})
    print("\nCombos:")
    print(combos.to_string(index=False))

    if dropped:
        print(f"\nDropped: {len(dropped)}")
        for name, reason in dropped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
