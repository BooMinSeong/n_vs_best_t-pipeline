# N vs best-T Pipeline

`maj@N` 정확도가 sample 예산 `N` 에 따라 어떻게 변하는지, 그리고 단일 temperature
선택 (T=0.1, 0.2, …, 1.2 중) 의 oracle best 가 다양한 deployable baseline
(T=1.0 fixed, T=0.1 fixed, random_T, equal_mix, consensus_vote) 대비 얼마나
이득인지를 시뮬레이션 기반으로 분석하는 파이프라인.

원본 코드는 `tts_analysis` (private) 의 `scripts/n_vs_best_t/` 에 있고, 이 repo 는
독립 배포/공유 목적의 **mirror** 입니다 (`git filter-repo` 로 history 보존 추출).
원본 commit reference: `c3d5909` (tts_analysis @ feat/auc-pilot-simulation).

## 디렉터리 구조

```
.
├── scripts/
│   ├── n_vs_best_t/           # 핵심 pipeline (run/merge/plot/build_report)
│   ├── cross_t_mode/
│   │   ├── lb_baselines.py    # load_aligned + equal_alloc + _share_with_ties
│   │   └── lb_algorithm_v1.py # cross-T LB routing (v1.0 backend)
│   └── slurm_common.sh        # SLURM 잡의 공통 env init
├── outputs/n_vs_best_t/       # 직전 sim 결과 + REPORT.md + figs/
└── pyproject.toml
```

## 의존성 설치

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

순수 `numpy + pandas + matplotlib` 만 필요.

## 입력 데이터 마련

이 repo 는 **raw distributions 데이터를 포함하지 않습니다** (~3.1GB).
입력은 `outputs/distributions-{dataset_year}_n256_{model}/distributions.json`
형식의 파일들로, 각 파일은 per-problem 의 12개 temperature × top-K answer
empirical 분포를 담고 있습니다 (`load_aligned`, `cross_t_mode/lb_baselines.py:49`
참조).

**해결 1 — symlink** (기존 환경에 데이터가 이미 있는 경우):

```bash
for d in /path/to/your/distributions-*_n256_*; do
    ln -s "$d" "outputs/$(basename "$d")"
done
```

**해결 2 — preprocessing pipeline 별도 실행**: 원본 `tts_analysis` repo 의
`analysis/preprocess.py` 가 HuggingFace Hub (`ENSEONG/preprocessed-...`)
에서 raw generation 결과를 다운로드 → per-temperature MV count 집계 →
`distributions.json` 으로 직렬화합니다.

## 실행 흐름

(1) **per-combo sim** (~30s–2h per combo, SLURM 권장):

```bash
sbatch scripts/n_vs_best_t/run_sim_v2_combo.slurm <DATASET> <MODEL>
# 예: sbatch scripts/n_vs_best_t/run_sim_v2_combo.slurm aime2023 Qwen2.5-3B
```

`run_sim_v2_combo.slurm` 는 generic launcher. SLURM 환경이 없다면 직접:

```bash
uv run python scripts/n_vs_best_t/run_sim_v2_one.py \
    --dist-path outputs/distributions-aime2023_n256_Qwen2.5-3B/distributions.json \
    --combo-name aime2023_Qwen2.5-3B \
    --out-csv outputs/n_vs_best_t/sim_v2/aime2023_Qwen2.5-3B.csv
```

산출: `outputs/n_vs_best_t/sim_v2/{combo}.csv` (long format: combo, baseline, N,
stratum, mean, std6, n_pids — baseline ∈ {T0.1..T1.2, random_T, equal_mix,
consensus_vote}).

(2) **merge → best-T 산정 → wide table**:

```bash
uv run python scripts/n_vs_best_t/merge_sim_v2.py
```

산출: `outputs/n_vs_best_t/best_t_table.csv` (wide format; per (model, dataset,
stratum, N) 의 best-T = 12 single-T 중 argmax + 5 mix baseline accuracy + gap).

(3) **그래프 재생성**:

```bash
uv run python scripts/n_vs_best_t/plot_curves.py
```

산출: `outputs/n_vs_best_t/figs/{model}__{dataset}/B_5curve_*.png`,
`C_gap_*.png`, `A_t_star_trajectory.png` 등.

(4) **보고서 빌드**:

```bash
uv run python scripts/n_vs_best_t/build_report.py
```

산출: `outputs/n_vs_best_t/REPORT.md` (Coverage / Headline / Per-combo /
Cross-comparison 섹션 포함).

## 주요 개념

- **best-T (oracle ceiling)**: 각 `(model, dataset, stratum, N)` 조합에 대해
  12개 single-T MV accuracy 중 argmax. 사후적으로 최적 T 를 알고 있다는
  가정의 ceiling — deployable baseline 의 upper bound.
- **stratum**: `L1`–`L5` 는 baseline solve-rate 기반 난이도 (L1 easiest);
  `Lr` 은 level=None (정답이 거의 안 나오는 unsolvable bucket); `overall` 은
  L1–L5+Lr 가중평균; `overall_md_compat` 은 L1–L5 만 (옛 markdown 보고서
  호환).
- **문제 제외 기준** (`load_aligned`): 모든 T 에서 답 추출 실패 또는 canonical
  정답이 분포에 한 번도 등장하지 않은 문제는 sim 입력에서 drop.
- **consensus_vote**: 각 T 의 MV winner 중에서 plurality vote. `equal_mix`
  (모든 T 의 답을 pool 해 한 번에 MV) 와는 N>12 일 때 진정한 알고리즘적 차이.

자세한 정의 / 결과 / 검증은 `outputs/n_vs_best_t/REPORT.md` 참고.

## Per-problem regret 분석 (`scripts/per_problem_regret/`)

Dataset-level 비교를 **per-problem 분포 분석**으로 확장한다 (oracle gap / regret 분포 /
$T^*(p)$ 분포 / paired 비교 / stochastic dominance). GPU 비용 0 — 같은
`distributions.json` pool 을 재사용하는 CPU 후처리.

**중요 (방법론)**: 입력은 raw generation pool 이 아니라 per-problem × 12 temperature 의
*empirical answer 분포* (`distributions.json`, seed 6 × 256 = 1536 generation 집계). 따라서
per-problem maj@N 정확도는 `multinomial(N, P)` 샘플링으로 추정한다 (기존 `run_sim_v2_one.py`
와 동일 엔진 재사용 → dataset-level 수치와 같은 통화). spec §2.1 의 "raw generation
bootstrap" 은 데이터 구조상 적용 불가하며 본 구현이 유일하게 가능한 방식이다.

```bash
# 한 combo end-to-end (compute → analyze → figures → REPORT)
uv run python scripts/per_problem_regret/run_pipeline.py \
    --combos aime2025_Qwen3-4B-Instruct-2507 --n-sims 5000

# 전체 combo (combos.txt). 대용량(mathfull 12500문제)은 SLURM 권장.
uv run python scripts/per_problem_regret/run_pipeline.py

# 이미 a_hat parquet 이 있으면 분석/그림/리포트만 재생성
uv run python scripts/per_problem_regret/run_pipeline.py --combos <C> --skip-compute

# acceptance / sanity 체크 (spec §1.2, §5)
uv run python scripts/per_problem_regret/validate.py
```

산출 (`outputs/per_problem_regret/`): `a_hat_per_problem__{combo}.parquet`,
`per_problem_oracle.parquet`, `per_problem_regret.parquet`, `summary_*.csv` (6종),
`REPORT.md` (최상위 인덱스: 방법론 + oracle-gap 매트릭스 + cross-combo overview +
combo별 목차 링크) + `reports/{model}__{dataset}.md` (combo당 1파일, N×전략 cross-table),
`figs/` (6종 figure, png+pdf). 모듈: `config` / `compute_a_hat` / `analyze` / `plots` /
`build_report` / `validate` / `run_pipeline`.

## SLURM 환경 가정

`scripts/slurm_common.sh` 가 다음을 수행:
- `module load gnu12/12.2.0`, `module load cuda/12.6` (CPU 잡이지만 venv 설정용)
- `source .venv/bin/activate`
- compiler/cache 환경변수 설정

이 환경 변수와 module system 이 없으면 SLURM 스크립트는 작동 안 함.
로컬 단일 머신에서는 위 (1) 의 `uv run python ...` 직접 실행으로 대체.

## License

(추후 추가)
