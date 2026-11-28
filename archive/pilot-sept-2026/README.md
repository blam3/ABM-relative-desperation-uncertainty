# Pilot simulations (September 2026)

Reduced-scale runs used to check the corrections in `CHANGELOG.md` before spending
cluster time on the full sweep. Every number in `stats.json` comes from these runs,
and the figures in `figures/` are built from it.

**These are previews, not results.** The grid is 50 x 50 (2,500 agents) with 4 to 8
replications per condition, against 100 x 100 and 100 replications in the real
sweep. Report the full sweep.

## What was run

| Study | Settings | Replications |
|---|---|---|
| `legacy` | June 2026 settings (`weighting="prelec_raw"`, t-distributed alpha clipped at 0.1) | 8 |
| `betafix` | beta on the log scale, June alpha draws | 4 |
| `fullfix` | beta on the log scale, log-normal alpha | 8 |
| `ineq` | corrected model, mean-preserving spreads of initial wealth x desperation rule | 6 |
| `traj` | 50-step trajectories, corrected model (beta = 0) and June model (beta = 1) | 6 |

## Files

- `pilot.py <study> <reps>` runs one study (`legacy`, `betafix`, `fullfix`, `ineq`).
- `traj_pilot.py` runs the 50-step trajectories.
- `fill_missing.py` reruns four runs lost when a worker process ran out of memory.
- `equiv_test.py` checks that the current code in legacy mode reproduces
  `../archive/june-2026-model/` exactly.
- `analyze.py <study>` prints a summary; `stats.py` writes `stats.json`.
- `make_figures.py` builds `figures/` from `stats.json` and `results_traj.csv`.
- `results_*.csv` hold one row per run (`results_traj.csv`: one row per run and step).

## What the pilot showed

1. With the June settings, the 50 x 50 pilot reproduces the published marginal
   means and regression coefficients to two or three decimals (`fig2`).
2. Correcting the weighting function reverses the direction of the likelihood
   insensitivity effect at every level of uncertainty aversion, and the
   beta x alpha interaction nearly disappears (`fig3`).
3. Offending rises over the run and then levels off while inequality keeps
   growing (`fig4`), because a desperate agent is most tempted when a successful
   theft would still close the gap (`fig6`).
4. Manipulating initial inequality directly raises offending at first, but the
   conditions converge as apprehension penalties generate their own inequality
   (`fig5`).

## Reproduce

```bash
pip install -r ../requirements.txt
python pilot.py fullfix 8
python stats.py
python make_figures.py
```
