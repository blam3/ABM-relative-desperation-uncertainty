# Running the sweep on Bouchet

Step-by-step for Yale's Bouchet cluster ([YCRC docs](https://docs.ycrc.yale.edu/clusters/bouchet/)).
Everything below assumes this repository is the working directory.

## 0. What runs where

| | Path | Why |
|---|---|---|
| Code | `~/project/dmurd` (or `~/dmurd`) | Small, backed up |
| Output | `$HOME/scratch/dmurd` | 10 TiB per group, **purged after 30 days**, not backed up |
| Results you keep | copy summaries and `analysis/` to `~/project` | 4 TiB per group, backed up |

Bouchet's filesystems are `/nfs/roberts/home`, `/nfs/roberts/project`, and
`/nfs/roberts/scratch`. Check what your home directory links to with
`ls -l $HOME | grep -iE 'scratch|project'`. If the scratch symlink has a different
name, set it once before submitting:

```bash
export DMURD_RESULTS_ROOT=/nfs/roberts/scratch/<your_group>/<netid>/dmurd
```

Output lands in four directories, because the supplementary scripts write beside
the main sweep rather than inside it:

| Directory under `$DMURD_RESULTS_ROOT` | Written by |
|---|---|
| `final_batch_run_v2/` | `batch_run_hpc.py` (main sweep) |
| `sensitivity_softmax_v2/` | `batch_run_sensitivity.py` tasks 1-9 |
| `sensitivity_convergence_v2/` | `batch_run_sensitivity.py` tasks 10-18 |
| `inequality_sweep/` | `batch_run_inequality.py` |

Step 6 aggregates all four; do not aggregate only the first one.

## 1. Get the code onto the cluster

If the repository is on GitHub:

```bash
ssh netid@bouchet.ycrc.yale.edu
mkdir -p ~/project && cd ~/project
git clone https://github.com/blam3/ABM-poverty-uncertainty.git dmurd
cd dmurd
```

Otherwise copy it from your laptop (skips the archive folders and your local venv):

```bash
rsync -avz --progress \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'deprecated' --exclude 'docs' --exclude 'results' --exclude 'analysis' \
  ~/Documents/GitHub/ABM-poverty-uncertainty/ \
  netid@bouchet.ycrc.yale.edu:~/project/dmurd/
```

Then create the log directory once, before any `sbatch`. Slurm opens
`logs/sweep-%A_%a.out` before the job script runs, so a missing `logs/` fails the
job no matter what `slurm/common.sh` does:

```bash
cd ~/project/dmurd && mkdir -p logs
```

## 2. Build the conda environment

Conda environments cannot be created on the login node, so grab an interactive
node first:

```bash
salloc -p devel -c 4 -t 1:00:00
module reset
module load miniconda
conda env create -f environment.yml     # creates the 'dmurd' environment
conda activate dmurd
python check_environment.py             # versions, a small run, seed reproducibility
exit                                    # release the interactive job
```

`check_environment.py` fails loudly if Mesa is older than 3.x or if the weighting
function is not the corrected one, so run it before anything else.

## 3. Smoke test (about five minutes)

```bash
sbatch slurm/smoke_test.sh
squeue -u $USER
cat logs/smoke-*.out
ls $HOME/scratch/dmurd/smoke_test
```

You should see one `summary_cond_21_*.csv`, one `cond_21_*.csv.gz`, and an
`analysis/` folder. If that works, the real sweep will too.

## 4. Main sweep

```bash
sbatch slurm/submit_main_sweep.sh
```

45 array tasks, one per condition, 24 cores each, at most 12 at a time. Each task
runs 100 replications of 10,000 agents for 25 steps: roughly 3 to 6 minutes per
condition, so the whole array usually finishes within an hour of starting.

Each condition writes two files to `$DMURD_RESULTS_DIR`:

| File | Size | Contents |
|---|---|---|
| `summary_cond_<n>_*.csv` | about 120 KB | One row per replication: offending at every step, the desperate and non-desperate rates, wealth percentiles, the linear-probability coefficients, the mean decision weight |
| `cond_<n>_*.csv.gz` | about 570 MB | One row per agent per step (gzipped) |

Across 45 conditions that is about 5 MB of summaries and 26 GB of per-agent rows
(83 GB before compression). To skip the per-agent rows entirely:

```bash
sbatch --export=ALL,DMURD_RAW=0 slurm/submit_main_sweep.sh
```

Monitor and check for failures:

```bash
squeue -u $USER
sacct -X -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
grep -L "JOB COMPLETE" logs/sweep-*.out     # tasks that did not finish
```

`grep -L` must print nothing. Then confirm every condition wrote its full 100
replications, because a task can be requeued mid-write and still look finished:

```bash
R="${DMURD_RESULTS_ROOT:-$HOME/scratch/dmurd}"
ls "$R"/final_batch_run_v2/summary_*.csv | wc -l      # expect 45
for f in "$R"/final_batch_run_v2/summary_*.csv; do
  n=$(( $(wc -l < "$f") - 1 ))
  [ "$n" -eq 100 ] || echo "$f has $n rows"
done
```

Rerun only the conditions that failed, for example tasks 3 and 17:

```bash
sbatch --array=3,17 slurm/submit_main_sweep.sh
```

## 5. Sensitivity and supplementary sweeps

```bash
sbatch slurm/submit_sensitivity.sh    # 18 tasks: softmax robustness, 50-step convergence
sbatch slurm/submit_inequality.sh     # Goal 3 / H4: exogenous inequality, alternative rule
```

Both write summaries only, and both can be submitted while the main array is still
queued: 18 x 24 plus the main array's 12 x 24 is 720 cores, inside the `day`
partition's 1,500 per user.

Neither writes into `final_batch_run_v2`. Both derive their output path from the
*parent* of `DMURD_RESULTS_DIR`, so the sensitivity summaries land in
`$DMURD_RESULTS_ROOT/sensitivity_softmax_v2` and
`$DMURD_RESULTS_ROOT/sensitivity_convergence_v2`, and the inequality sweep writes a
single `$DMURD_RESULTS_ROOT/inequality_sweep/summary.csv` (900 rows: 9 cells x 100
replications). Check them the same way:

```bash
grep -L "SENSITIVITY JOB COMPLETE" logs/sens-*.out    # expect no output
tail -1 logs/ineq-*.out                               # expect DONE
```

## 6. Build the analysis files

All four sweeps, in order. `aggregate_results.py` globs `summary_*.csv` in one
directory at a time, so each output directory needs its own call:

```bash
module load miniconda && conda activate dmurd
cd ~/project/dmurd
R="${DMURD_RESULTS_ROOT:-$HOME/scratch/dmurd}"

python aggregate_results.py --results "$R/final_batch_run_v2" \
  --out analysis --metamodel
python aggregate_results.py --results "$R/sensitivity_softmax_v2" \
  --out analysis/sensitivity_softmax
python aggregate_results.py --results "$R/sensitivity_convergence_v2" \
  --out analysis/sensitivity_convergence
```

The first call writes `analysis/replication_level.csv` (4,500 rows, the unit of
analysis), `analysis/condition_means.csv`, and `analysis/metamodel_terms.csv`, and
prints the OLS metamodel with eta squared next to partial eta squared. Note that
the alpha factor is coded as likelihood insensitivity, iota = -log2(alpha), so a
positive coefficient means insensitivity increases offending.

### The two supplements that need their own grouping

`condition_means.csv` groups by `Beta_Loc`, `Alpha_Loc` and `Threshold`. All nine
softmax cells share the same three values, so that file collapses to one row and
hides the factors that actually vary. Group the replication file by tau and theta
instead:

```bash
python -c "import pandas as pd; d=pd.read_csv('analysis/sensitivity_softmax/replication_level.csv'); print(d.groupby(['Tau_Softmax','Theta_Softmax']).Crime_Final.agg(['mean','std','count']).round(4))"
```

The inequality sweep writes its own row format rather than `run_summary.py`'s, so
`aggregate_results.py` does not read it at all. Read it directly:

```bash
python -c "import pandas as pd,os; d=pd.read_csv(os.path.expandvars('$R/inequality_sweep/summary.csv')); print(len(d),'rows'); print(d.groupby(['spread','rule','thresh'],dropna=False)[['init_gap','crime_1','crime_25','gap_25']].mean().round(4))"
```

Expect 900 rows and 9 cells. The convergence summaries carry `Crime_1` through
`Crime_50` instead of `Crime_25`, which is why they aggregate into their own
directory and cannot be concatenated with the main sweep.

## 7. Bring results back and protect them from the purge

```bash
# on the cluster: keep the small files somewhere backed up
R="${DMURD_RESULTS_ROOT:-$HOME/scratch/dmurd}"
mkdir -p ~/project/dmurd_results
cp "$R"/final_batch_run_v2/summary_*.csv ~/project/dmurd_results/
cp "$R"/inequality_sweep/summary.csv ~/project/dmurd_results/inequality_summary.csv
cp -r "$R"/sensitivity_softmax_v2 "$R"/sensitivity_convergence_v2 ~/project/dmurd_results/
cp -r analysis ~/project/dmurd_results/

# on your laptop
rsync -avz netid@bouchet.ycrc.yale.edu:~/project/dmurd_results/ \
  ~/Documents/GitHub/ABM-poverty-uncertainty/results_from_cluster/
```

Scratch is purged after 30 days, so either move the `.csv.gz` files to project
space (26 GB of the 4 TiB group quota) or accept that per-agent analyses would
need a rerun. The sweep is deterministic: the same code and root seed reproduce
the same runs.

## 8. Knobs

Every batch script reads these environment variables, so you can change a run
without editing code:

| Variable | Default | Effect |
|---|---|---|
| `DMURD_RESULTS_ROOT` | `$HOME/scratch/dmurd` | Where output goes |
| `DMURD_RAW` | `1` | `0` writes summaries only |
| `DMURD_GZIP` | `1` in the Slurm scripts | `0` writes plain CSV |
| `DMURD_REPLICATIONS` | `100` | Replications per condition |
| `DMURD_STEPS` | `25` | Steps per run |
| `DMURD_GRID` | `100` | Grid side; 30 makes a quick test |
| `DMURD_ROOT_SEED` | `20260915` | Change only for a genuinely new sweep |
| `DMURD_CONDA_ENV` | `dmurd` | Environment name |

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CondaError` when creating the environment | You are on the login node. Use `salloc -p devel -c 4 -t 1:00:00` first. |
| `ModuleNotFoundError: mesa` inside a job | The job did not activate the environment. `module load miniconda` and `conda activate dmurd` belong inside the script, not in `~/.bashrc`. |
| `CommandNotFoundError: conda activate` inside a job | The batch shell had no conda hook. `slurm/common.sh` now sources `$(conda info --base)/etc/profile.d/conda.sh` when the module does not define the shell function, so this should not recur; if it does, `module load miniconda` itself failed. |
| Job dies immediately, no `logs/*.out` file | `logs/` did not exist at submission time. `mkdir -p logs` and resubmit (step 1). |
| No `summary_*.csv` for the sensitivity or inequality runs in `final_batch_run_v2` | They are not written there. Look in `sensitivity_softmax_v2`, `sensitivity_convergence_v2` and `inequality_sweep` beside it (step 0). |
| Task killed, `State=OUT_OF_MEMORY` | Raise `--mem-per-cpu` to 6G, or lower `--cpus-per-task`. Each worker holds one 10,000-agent model plus the rows for one replication. |
| Array task hits the time limit | Resubmit only those tasks with `--array=<ids>` and a longer `--time`. The `week` partition allows 7 days but only 96 CPUs per user. |
| Jobs sit in the queue | 45 tasks x 24 cores exceeds a polite share; the script already throttles to 12 at a time with `%12`. Lower it, or use `--partition=scavenge` (preemptable, and the scripts set `--requeue`). |
| Disk quota errors mid-sweep | You are writing to home (125 GiB). Set `DMURD_RESULTS_ROOT` to scratch. |
| Results differ between runs | Check that `DMURD_ROOT_SEED` did not change and that each task wrote the `Seed` column; seeds come from `SeedSequence([root, condition, rep])`. |

## 10. Partition reference (Bouchet, September 2026)

| Partition | Max time | Notes |
|---|---|---|
| `devel` | 6 hours | 4 CPUs per user; use for setup and smoke tests |
| `day` | 24 hours | 1,500 CPUs per user; the default choice here |
| `week` | 7 days | 96 CPUs per user |
| `scavenge` | 24 hours | Preemptable, requeued automatically |

Source: [Bouchet cluster documentation](https://docs.ycrc.yale.edu/clusters/bouchet/).
