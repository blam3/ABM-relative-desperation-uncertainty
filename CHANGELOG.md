# Changelog

Reverse-chronological. Entries before 2026-05-15 were not kept in this repository;
the running log from the cluster copy is summarized under "Earlier".

---

## 2026-11-28 - Victimization arms (victim_mode) and synchronous schedule (sync)

Behind flags whose defaults reproduce tag `jasss-v1` (local tag at `6548381`).
With `victim_mode="none"` and `sync=False`, same-seed output is identical to
jasss-v1 in every pre-existing column, at full float precision.

- `model.py`: `victim_mode` in {"none", "global", "local"} and `sync`. Both are read
  from `DMURD_VICTIM` / `DMURD_SYNC` unless passed explicitly. New methods:
  `draw_victim`, `transfer`, `advance` (one step without data collection) and
  `victim_summary`. Victim draws come from a separate RNG.
- `agents.py`: `step()` is split into `decide()` (all random draws, in the jasss-v1
  order) and `settle()`, which applies income, penalty and living cost. The
  comparison sample is stored, and per-step accounting and victimization fields
  are logged.
- `run_summary.py`: new columns after `Victim_Mode`: `Sync`, `Victim_Rate_Final`,
  `Victim_Rate_Mean`, `Mean_Loss_Final`, `Mean_Transfer`, `Mean_Transfer_Final`
  and `Zero_Transfer_Share`.
- `batch_run_hpc.py`: `OptimizedModel.step` calls `advance()`, so sync mode is
  honored. Six raw columns are appended.
- All batch scripts log and record the `victim_mode` and `sync` that actually ran,
  replacing the hard-coded "none".
- New `tests/test_victimization.py` (with `tests/_trajectory_driver.py`) covers the
  jasss-v1 regression, wealth accounting, sampling and an opt-in smoke test.

---

## 2026-11-28 - Outputs record their own lambda and victim mode

Output-only change; model dynamics are unchanged (same-seed runs are byte-identical
in every pre-existing column and in the per-agent rows).

- `run_summary.py`: two new columns at the end of every summary row, `Lambda` and
  `Victim_Mode` (always `"none"`: no victimization logic exists yet).
- `batch_run_hpc.py`, `batch_run_sensitivity.py`, `batch_run_neighbor_sensitivity.py`
  and `batch_run_lambda_sweep.py` pass the lambda each run actually used;
  `batch_run_inequality.py` appends `lambda` and `victim_mode` to its rows.
- Every batch script prints lambda and victim mode in its start-up line, so the
  Slurm log records them too.
- New test: a summary row has one value per column and carries both fields.

---

## 2026-11-28 - Sync of the cluster working copy with GitHub

Merges the 2026-11-18 cluster work into the 2026-11-25 GitHub version. No model
dynamics changed: `agents.py`, `cognitive_functions.py`, `run_summary.py`,
`aggregate_results.py` and `batch_run_inequality.py` were already identical.

- lambda = 0.5 default (entry below) applied to `run.py`'s `simulate()` and its
  metadata as well; `examples/demo/` regenerated at lambda = 0.5.
- Environment pinned to the cluster environment that produced the research
  sweeps: Python 3.11, Mesa 3.3.1 (`environment.yml`, `requirements.txt`, CI).
  This replaces the 2026-11-25 Python 3.12 / Mesa 3.5.3 pins.
- Added `batch_run_lambda_sweep.py`, `batch_run_neighbor_sensitivity.py`,
  `analyze_lambda_sweep.py`, `analyze_for_manuscript.py`,
  `make_manuscript_figures.py`, `make_supplementary.py`, `notebooks/` and the
  reproducibility part of `archive/` (June 2026 model code, September 2026 pilot).
- Removed the lambda = 0.2 Slurm scripts `slurm/submit_main_sweep.sh` and
  `slurm/submit_sensitivity.sh`; use the `_lambda05` scripts.
- `submit_main_sweep_lambda05.sh` honours `DMURD_RAW=1` for per-agent rows
  (default still summaries only).
- Caveat on the 2026-11-18 entry: the lambda = 0.5 status of the September
  outputs rests on byte-identical reruns. The summary files themselves do not
  record lambda.

---

## 2026-11-25 - Presentation and reproducibility update

- Added an installable `requirements.txt`, small seeded CLI demo, optional plot
  and agent exports, run metadata, scientific boundary checks, and CI.
- Fixed `run.py` reproducibility: the old entry point supplied only a Mesa seed
  while NumPy/SciPy agent draws use the global NumPy RNG. The demo now seeds both.
  This changes the single-run random realization, not the scientific dynamics;
  it does not retroactively validate any prior sweep. Batch scripts are unchanged.
- Replaced missing pilot-file links with an explicitly labeled software example.
  Default `run.py` is now a small demonstration; the previous 100×100/25-step
  scale is available through explicit flags. Initialization is excluded from the
  demo plot, since all decisions are zero before the first step.
- Pin Mesa 3.5.3, checked during this cleanup. Mesa requires Python >=3.12;
  the Conda environment, README, and CI now use Python 3.12 after the initial
  CI run exposed the unsupported Python 3.11 combination. Make environment checks portable
  when CPU affinity is unavailable. No full research sweep was run.

---

## 2026-11-18 - lambda = 0.5 is now the default everywhere

Checked whether the sensitivity and inequality scripts read `DMURD_LAMBDA`. They do:
`batch_run_sensitivity.py`, `batch_run_neighbor_sensitivity.py` and
`batch_run_inequality.py` all copy `batch_run_hpc.MODEL_OPTIONS`, and none overrides
`lambd`. But `submit_inequality.sh` never set the variable, so the inequality sweep
ran at the code default of 0.2. The existing inequality results (job 26860827,
2026-09-20, analyzed in `analysis/inequality_sweep/`) are therefore lambda = 0.2.

- `submit_inequality.sh` now exports `DMURD_LAMBDA=0.5`.
- Default lambda changed from 0.2 to 0.5 in `batch_run_hpc.py` (`MODEL_OPTIONS`),
  `model.py` (`rel_DMAP_model`) and `run.py`, so a script that does not set it can
  no longer fall back to 0.2. `DMURD_LAMBDA=0.2` still overrides.
- Archived reproductions that relied on the old default now pass `lambd=0.2`
  explicitly (`archive/pilot-sept-2026/{pilot,traj_pilot,equiv_test}.py`, and the
  instructions in `archive/june-2026-model/README.md`). `equiv_test.py` still reports
  the current code identical to the June code under the June settings.
- `batch_run_lambda_sweep.py` is unaffected: it sets lambda per condition.
- Reran the three robustness analyses at lambda = 0.5 (jobs 33839586 softmax +
  50-step convergence, 33839587 comparison-group size; output in
  `~/scratch/dmurd/rerun-2026-11-18/`). All 93 summary files are identical to the
  2026-09-29 outputs in `~/scratch/dmurd/*_lambda0.5/` (max absolute difference 0),
  so those September outputs, and the supplement rebuilt from them on Oct 4, are
  lambda = 0.5 results.

---

## 2026-09-29 - Desperation-severity (lambda) sweep and recalibration

Under the previous lambda = 0.2, the per-capita offending rate among desperate
agents *fell* at high nu (see Fig 2 in the 2026-09-18 revision). Because ν is a
rank threshold, enlarging it pulls more marginally-poor agents into the desperate
label; their upward wealth gap D is small, so `lambda * D` barely shifts their
reference point and they behave nearly like non-desperate agents. The group mean
gets diluted. This contradicted the pre-registered prediction that offending
among the desperate should rise with the desperation threshold.

### `batch_run_lambda_sweep.py`, `slurm/submit_lambda_sweep.sh`, `analyze_lambda_sweep.py`

Five-level sweep of lambda in {.05, .1, .2, .5, 1.0} crossed with the five nu
levels, fixed at the no-distortion cognitive centre (beta = 0, alpha = 1),
100 replications per cell. The desperate-agent offending rate rises monotonically
with nu starting at lambda >= 0.5 (see the table in `PARAMETERS.md` Section 6).

### `batch_run_hpc.py`

`MODEL_OPTIONS["lambd"]` now reads `DMURD_LAMBDA` if set. Default unchanged at 0.2
for reproducibility of the 2026-09-18 sweep.

### `slurm/submit_main_sweep_lambda05.sh`

New submission script: reruns the 45-condition main sweep with `DMURD_LAMBDA=0.5`
into `RESULTS_ROOT/final_batch_run_lambda0.5/`. Summaries only. 4,500 replications.

### `make_manuscript_figures.py`

`SCRATCH` default now points at the lambda = 0.5 sweep. Override with
`DMURD_FIGURE_DATA` to regenerate against any other summary directory. Same for
`DMURD_FIGURE_OUT`.

### `analysis/figures/`

Overwritten with the lambda = 0.5 figures. Every panel that involves nu now shows
a clean monotone effect. The desperation gap magnitudes are smaller (~.02-.03,
down from ~.06 pooled), but they are consistent across nu rather than reversing.
The old lambda = 0.2 figures can be regenerated with:

    DMURD_FIGURE_DATA=$HOME/scratch/dmurd/final_batch_run_v2 \
        python make_manuscript_figures.py

### Results section

`docs/results_section_revised.md` rewritten to reflect the new sweep (all means,
CIs, standardized regression coefficients, and the response-surface extremes).

### PARAMETERS.md

Section 6 expanded with the lambda sweep table and the rationale for lambda = 0.5;
Section 1 table and Section 8 updated to match.

---

## 2026-09-18 - Cluster support (Bouchet)

Everything needed to upload this directory to Yale's Bouchet cluster and run the
full sweep.

### Added

- `CLUSTER.md`: transfer, conda setup, smoke test, the three sweeps, aggregation,
  bringing results back, a troubleshooting table, and Bouchet's partition limits.
- `slurm/`: `submit_main_sweep.sh` (45-task array, 24 cores per task, throttled to
  12 at a time), `submit_sensitivity.sh` (18 tasks), `submit_inequality.sh`,
  `smoke_test.sh` (devel partition), and `common.sh` (module load, conda activate,
  one BLAS thread per process, scratch paths).
- `environment.yml` (conda) and `check_environment.py`, which verifies package
  versions, that the weighting function is the corrected one, and that a seed
  reproduces a run.
- `run_summary.py`: the replication-level summary written by every batch script.
  One row per run with offending at each step, the desperate and non-desperate
  rates, the share desperate, wealth percentiles, the share in debt, the
  linear-probability coefficients, and the mean decision weight.
- `aggregate_results.py`: concatenates the summaries into
  `analysis/replication_level.csv` and `analysis/condition_means.csv`, and with
  `--metamodel` fits the OLS metamodel with eta squared (which sums to R squared)
  beside partial eta squared, coding alpha as insensitivity, iota = -log2(alpha).

### Changed

- `batch_run_hpc.py`: always writes the summary file; per-agent rows are optional
  (`DMURD_RAW`) and can be gzipped (`DMURD_GZIP`); output directory, replications,
  steps, grid size, and root seed are read from the environment. Per-agent values
  are rounded (wealth to 2 decimals, rank to 4, deprivation to 2) and position is
  written as `x;y`, which brings the sweep from about 110 GB to 83 GB, or 26 GB
  gzipped.
- `batch_run_sensitivity.py` and `batch_run_inequality.py` follow the same
  environment settings and write summaries only.
- `.gitignore`: `deprecated/` and `docs/` are kept on disk but no longer tracked,
  so the manuscript drafts and the 58 MB archive are not published to GitHub.
  `analysis/` and `results_from_cluster/` are ignored as well.

### Note

`deprecated/` and `docs/` were committed in the reorganization commit. Untrack them
with `git rm -r --cached deprecated docs` (the files stay on disk) before pushing.

---

## 2026-09-18 - Repository reorganization

The working folder had drifted: the code at the root was the November 2025
version, the code that produced the June 2026 manuscript results lived only on the
cluster, and exploratory notebooks from 2025 sat next to current work. This
release makes the root the current model and files everything else by status.
**Nothing was deleted.** Files that are no longer current were moved to
`deprecated/`.

### New structure

```
.                          current model code (run this)
  README.md  CHANGELOG.md  PARAMETERS.md  requirements.txt  .gitignore
  cognitive_functions.py  agents.py  model.py  run.py
  batch_run_hpc.py  batch_run_sensitivity.py  batch_run_inequality.py
pilot/                     September 2026 reduced-scale runs: code, results, figures
legacy/june-2026-hpc/      the code behind the June 2026 manuscript results
docs/                      manuscript-side documents
notebooks/                 current diagnostics (June 2026)
deprecated/                superseded code, notebooks, and outputs (kept, not deleted)
```

### Moved

| From | To |
|---|---|
| `cognitive_functions.py` (Nov 2025), `test_run_July_23_25.py` | `deprecated/code/` |
| `dmap/` (package version of the model) | `deprecated/dmap-package-2025/` |
| `misc/` | `deprecated/misc-2025/` |
| 14 exploratory notebooks (`01-` to `07-`, `DMAP.ipynb`, `absolute-threshold.ipynb`, the description notebooks, `three_choice_variation.ipynb`, `documenting-progress.ipynb`) | `deprecated/notebooks/` |
| `wealth-over-time.png`, `proportion-rule-breaking-cost-rb-is500.png`, `model_data.csv`, `batch_run_4_results_safe.csv`, `three_way_interaction_summary.csv` | `deprecated/outputs/` |
| `README.md` (June 2025) | `deprecated/documents/README-2025-06.md` |
| Word lock files (`~$...docx`) | `deprecated/documents/` |
| `Methods draft - ABM poverty.docx` | `docs/` |
| `gini_trajectory_diagnostic.ipynb`, `diagnostic-trajectories*.png` | `notebooks/` |

### Added

- The revised model and sweep scripts at the root (see the 2026-09-15 entry).
- `PARAMETERS.md`: every parameter value, the reason for it, and what the three
  levels of uncertainty aversion now mean.
- `pilot/`: the September 2026 pilot code, raw results, `stats.json`, and six
  figures in `pilot/figures/`.
- `legacy/june-2026-hpc/`: the June 2026 model, copied from the manuscript project
  and verified to reproduce the pilot reference model agent for agent.
- `docs/`: the recommendations document and the discussion outline.
- `requirements.txt` and `.gitignore` (ignores `.venv/`, `__pycache__/`, `.DS_Store`,
  `results/`, and Word lock files; the pilot results are committed on purpose).

---

## 2026-09-15 - Weighting-function, sampling, and seeding corrections

All three change results. The 4,500-run sweep has to be rerun before the
manuscript's Results can be updated; `results/final_batch_run_v2/` is the
destination, so the June output is not overwritten.

### `cognitive_functions.py` - Prelec elevation parameter entered on the wrong scale (Critical)

**Problem:** `prelec()` computed w(p) = exp(-beta (-ln p)^alpha) with beta used
directly as Prelec's elevation parameter, which must be positive. At the design
value beta = 0, w(p) = 1 for every p, so agents treated a successful theft as
certain. At beta = -1, w(p) > 1, so the apprehension outcome received a negative
weight and raised the value of offending. With the sweep's beta SD of 0.5, about
98% of agents in the beta = -1 conditions, 50% in the beta = 0 conditions, and 2%
in the beta = 1 conditions had w(p) > 1. The no-distortion case was beta = 1,
alpha = 1, not beta = 0.

**Fix:** new `weighting` argument. The default, `"prelec_log_elevation"`, uses
delta = exp(beta), so beta = 0 is the no-distortion baseline, beta > 0 is
uncertainty aversion, beta < 0 is uncertainty seeking, and w(p) stays inside
(0, 1). `weighting="prelec_raw"` reproduces the June results exactly.

### `agents.py` - likelihood-sensitivity draws piled up at the clipping bound

**Problem:** alpha was drawn from t(df = 9999, scale = 0.5) and clipped to
[0.1, 10]. In the alpha = 0.5 conditions, 21% of agents were set to exactly 0.1
(4% at alpha = 1).

**Fix:** new `alpha_dist` argument. The default, `"lognormal"`, draws
alpha = alpha_loc * exp(0.35 z). `alpha_dist="t_clip"` with `alpha_scale=0.5`
reproduces the June draws.

### `batch_run_hpc.py`, `batch_run_sensitivity.py` - seeds reused across conditions

**Problem:** seed = 12345 + rep + int(|beta|*100 + alpha*10 + nu*1000) produced
815 distinct seeds for 4,500 runs, gave the beta = -1 and beta = +1 conditions
identical random streams, and let runs in different conditions share seeds.

**Fix:** seeds come from `numpy.random.SeedSequence([ROOT_SEED, condition, rep])`
and are written to the output (`Seed` column).

### Other changes

- `model.py`: removed the unused multiplicative desperation path
  (`use_refshift=False`); the reference-shift mechanism is the only one. Default
  `beta_loc` is 0.
- `agents.py`: new `desperation_rule` (`"rank"` default, `"median_ratio"` optional
  with `kappa`) and `pareto_shape` / `pareto_scale` arguments.
- `batch_run_hpc.py`: writes to `results/final_batch_run_v2/`; adds `Seed`,
  `WealthStart`, `RelDeprivation`, per-agent `Beta` and `Alpha`, and `InitGap`.
- `batch_run_sensitivity.py`: softmax sweep centred on beta = 0, alpha = 1;
  writes to `*_v2` folders.
- New `batch_run_inequality.py`: mean-preserving spreads of initial wealth
  (initial Gini about .08, .30, .52) crossed with the rank and median-ratio
  desperation rules, for Goal 3 and H4.

### Verification

- In legacy mode the revised code reproduced the June code agent for agent
  (identical wealth, decisions, beta and alpha after 10 steps on a 30 x 30 grid
  with the same seed).
- A 50 x 50 pilot of the June settings reproduced the published marginal means and
  regression coefficients to two or three decimals (`pilot/`).

### Still to do

```bash
mkdir -p logs                           # Slurm needs this before the first submit
sbatch --export=ALL,DMURD_RAW=0 slurm/submit_main_sweep.sh   # 45 conditions x 100 reps
sbatch slurm/submit_sensitivity.sh      # 18 tasks -> sensitivity_*_v2/
sbatch slurm/submit_inequality.sh       # Goal 3 / H4 -> inequality_sweep/summary.csv
```
Then aggregate all four output directories, which is `CLUSTER.md` step 6: the
sensitivity and inequality scripts write beside `final_batch_run_v2`, not inside
it, so one `aggregate_results.py` call does not pick them up. `DMURD_RAW=0` skips
the 26 GB of per-agent rows; nothing in the analysis reads them.

Recode alpha for reporting as likelihood insensitivity, iota = -log2(alpha), and
report eta squared alongside partial eta squared.

---

## 2026-06-03 - Sign correction in the subjective-value functions

The three subjective-value functions for offending used
`w(p) * u(success) - (1 - w(p)) * u(failure)`, which gives the wrong comparative
statics: larger fines made offending more attractive. Corrected to the standard
binary-lottery form `w(p) * u(success) + (1 - w(p)) * u(failure)`. The full sweep
was rerun after this fix; every number in the June 2026 manuscript comes from the
corrected code.

---

## 2026-05-15 - Audit corrections (summary)

From the cluster changelog:

- `agents.py`: the neighbor search passed `num_neighbors` (20) as the Moore
  radius, covering 1,680 cells instead of a local group. Fixed to radius 3 with a
  random sample of 20, which changed agent behavior and required a new sweep.
- `model.py`: activation changed to `shuffle_do` so the base model matches the
  batch-run model.
- `batch_run_hpc.py`: NumPy's global RNG was not seeded per worker, so beta and
  alpha draws were not reproducible.
- Added `batch_run_sensitivity.py` (softmax robustness and a 50-step convergence
  check) and identifiability plots.
- Noted, and not acted on at the time: beta = 0 sets w(p) = 1 for all p and is not
  a neutral baseline. Fixed on 2026-09-15.

---

## Earlier

Model development from May 2025 to April 2026 is recorded in the notebooks now in
`deprecated/notebooks/` and in the commit history.
