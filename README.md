# Relative desperation and uncertainty: an agent-based model of property crime

An agent-based model of offending in which the decision to steal depends on two
things: where an agent stands relative to its neighbors, and how it evaluates
uncertain outcomes. The model combines a relative desperation threshold
(relative deprivation theory) with Prelec's probability weighting function
(behavioral economics) inside a spatial population of 10,000 agents.

Each step, every agent chooses between law-abiding behavior, which pays a certain
wage, and offending, which pays more when it succeeds and costs a large penalty
when it does not. Offending has a negative expected payoff by design, so any
offending the model produces comes from how agents weigh the odds and from how far
behind their neighbors they have fallen.

This repository accompanies a manuscript under preparation for *Social
Psychological and Personality Science*.

## Layout

```
cognitive_functions.py     utility, probability weighting, choice rule
agents.py                  agent: comparison group, desperation, decision, wealth update
model.py                   Mesa model: grid, scheduling, data collection
run.py                     one validation run -> model_data.csv, agent_data.csv
batch_run_hpc.py           main sweep: 3 beta x 3 alpha x 5 thresholds x 100 reps
batch_run_sensitivity.py   softmax robustness and a 50-step convergence check
batch_run_inequality.py    exogenous inequality and an alternative desperation rule
run_summary.py             the replication-level summary every batch script writes
aggregate_results.py       builds analysis/replication_level.csv and the metamodel
check_environment.py       preflight check to run before submitting a sweep
environment.yml            conda environment for the cluster
slurm/                     Slurm submission scripts for Bouchet
CLUSTER.md                 how to run the sweep on Yale's Bouchet cluster
PARAMETERS.md              every parameter value and the reason for it
CHANGELOG.md               what changed, when, and why
pilot/                     September 2026 pilot: code, raw results, figures
legacy/june-2026-hpc/      the code behind the June 2026 manuscript results
docs/                      manuscript-side documents
notebooks/                 current diagnostics
deprecated/                superseded code, notebooks, and outputs (kept for reference)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python check_environment.py      # versions, a small run, seed reproducibility
python run.py                    # one 100 x 100 run, 25 steps
python batch_run_hpc.py          # all 45 conditions locally (slow; use the cluster)
python batch_run_inequality.py   # exogenous-inequality supplement
```

Results go to `results/` by default, or wherever `DMURD_RESULTS_DIR` points, and
are not tracked by git.

## Run it on the cluster

`CLUSTER.md` is the step-by-step guide for Yale's Bouchet cluster: transfer, conda
environment, a five-minute smoke test, then

```bash
sbatch slurm/submit_main_sweep.sh     # 45 conditions x 100 replications
sbatch slurm/submit_sensitivity.sh    # softmax robustness and convergence
sbatch slurm/submit_inequality.sh     # Goal 3 / H4 supplement
python aggregate_results.py --results $HOME/scratch/dmurd/final_batch_run_v2 \
       --out analysis --metamodel
```

Every run writes a replication-level summary row (about 5 MB for the whole sweep);
the per-agent rows are optional and gzipped (about 26 GB), so they belong on
scratch.

## The three uncertainty conditions

The weighting function is w(p) = exp(-exp(beta) (-ln p)^alpha). With a successful
theft at p = .80 and alpha = 1:

| Condition | beta | Weight on a successful theft | Weight on apprehension (objective .20) |
|---|---|---|---|
| Uncertainty seeking | -1 | .92 | .08 |
| No distortion | 0 | .80 | .20 |
| Uncertainty averse | +1 | .55 | .45 |

`PARAMETERS.md` explains why beta enters as an exponent, what alpha does.

## Pilot runs

`pilot/` holds reduced-scale runs (50 x 50 grid) used to check the corrections
before committing cluster time. With the June settings the pilot reproduces the
published results closely; with the corrected code three conclusions change.
Figures are in `pilot/figures/`:

| Figure | Shows |
|---|---|
| `fig1_weighting_function.png` | Decision weights before and after the fix |
| `fig2_pilot_validation.png` | Pilot vs. published results, June settings |
| `fig3_beta_by_alpha.png` | Likelihood insensitivity reverses direction |
| `fig4_trajectories.png` | Offending levels off while inequality keeps growing |
| `fig5_exogenous_inequality.png` | Initial inequality raises offending, then conditions converge |
| `fig6_desperation_gap_curve.png` | Why offending peaks at a moderate wealth gap |

Pilot numbers are previews. Report results from the full sweep.

## Background reading

- Prelec, D. (1998). The probability weighting function. *Econometrica, 66*(3), 497-527. https://doi.org/10.2307/2998573
- de Courson, B., & Nettle, D. (2021). Why do inequality and deprivation produce high crime and low trust? *Scientific Reports, 11*, 1937. https://doi.org/10.1038/s41598-020-80897-8
- Radkani, S., Holton, E., de Courson, B., Saxe, R., & Nettle, D. (2023). Desperation and inequality increase stealing. *Royal Society Open Science, 10*, 221385. https://doi.org/10.1098/rsos.221385
- Mishra, S., Barclay, P., & Sparks, A. (2017). The relative state model. *Personality and Social Psychology Review, 21*(2), 176-198. https://doi.org/10.1177/1088868316644094
- Gonzalez-Jimenez, V. (2026). Poverty and biased evaluations of uncertainty: Theory and experimental evidence.
- Jackson, J. C., Rand, D., Lewis, K., Norton, M. I., & Gray, K. (2017). Agent-based modeling: A guide for social psychologists. *Social Psychological and Personality Science, 8*(4), 387-395. https://doi.org/10.1177/1948550617691100

## Citation

```
Lam, B. (2026). An agent-based model of relative desperation and uncertainty in
property crime [Computer software]. https://github.com/blam3/ABM-poverty-uncertainty
```
