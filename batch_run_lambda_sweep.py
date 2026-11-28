"""
Lambda sweep: does raising the reference-shift multiplier lambda make the
per-capita offending rate among desperate agents rise monotonically with the
desperation threshold nu?

Design
------
Fixed at the no-distortion cognitive baseline (beta = 0, alpha = 1) so any
change in the desperate-agent rate reflects the reference-shift mechanism
rather than probability-weighting effects.

    lambd     in {0.05, 0.1, 0.2, 0.5, 1.0}   (0.5 is the default since 2026-11-18)
    threshold in {0.05, 0.20, 0.35, 0.50, 0.65}

5 x 5 = 25 conditions, 100 replications each, 25 steps, 100x100 grid.

Output: DMURD_RESULTS_DIR (default: scratch/dmurd)/lambda_sweep/
        one summary_*.csv per condition (no per-agent rows).

Usage:  sbatch slurm/submit_lambda_sweep.sh   or   python batch_run_lambda_sweep.py
"""
import sys
import os
import csv
import gc
import itertools
import time
import numpy as np
from multiprocessing import Pool, cpu_count

import batch_run_hpc as main_sweep
from batch_run_hpc import OptimizedModel, MODEL_OPTIONS, ROOT_SEED, SAFE_CORE_LIMIT
import run_summary as rs

N_REPLICATIONS = main_sweep.N_REPLICATIONS
MAX_STEPS = main_sweep.MAX_STEPS
BASE = os.path.dirname(main_sweep.RUN_FOLDER) or "results"
OUT_DIR = os.path.join(BASE, "lambda_sweep")

LAMBDA_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]
THRESHOLD_VALUES = [0.05, 0.20, 0.35, 0.50, 0.65]
CONDITIONS = list(itertools.product(LAMBDA_VALUES, THRESHOLD_VALUES))

FIXED_BETA = 0
FIXED_ALPHA = 1


def seed_for(task_id, rep_id):
    # 3000 offset keeps this sweep's seed stream disjoint from main and softmax sweeps.
    return int(np.random.SeedSequence([ROOT_SEED, 3000 + task_id, rep_id]).generate_state(1)[0])


def worker(args):
    task_id, rep_id, lam, thresh = args
    seed = seed_for(task_id, rep_id)
    np.random.seed(seed)
    opts = dict(MODEL_OPTIONS)
    opts.update(lambd=lam)
    model = OptimizedModel(beta_loc=FIXED_BETA, alpha_loc=FIXED_ALPHA,
                           income_rank_threshold=thresh, seed=seed, **opts)
    agents = list(model.agents)
    init_gap = rs.gap([a.wealth for a in agents])
    crime, desperate_share = [], []
    try:
        for _ in range(MAX_STEPS):
            model.step()
            c, s = rs.step_stats(agents)
            crime.append(c)
            desperate_share.append(s)
        meta = (task_id, rep_id, seed, FIXED_BETA, FIXED_ALPHA, thresh,
                MODEL_OPTIONS["tau_softmax"], MODEL_OPTIONS["theta_softmax"],
                MODEL_OPTIONS["desperation_rule"])
        summary = rs.summarize(agents, crime, desperate_share, meta, init_gap, lambd=lam)
    finally:
        del model
        gc.collect()
    return summary


def run_condition(idx, n_cores):
    lam, thresh = CONDITIONS[idx]
    tid = idx + 1
    path = os.path.join(OUT_DIR, f"summary_lambda{lam}_thresh{thresh}.csv")
    tasks = [(tid, r, lam, thresh) for r in range(N_REPLICATIONS)]
    t0 = time.time()
    print(f"Task {tid}/{len(CONDITIONS)}: lambd={lam}, threshold={thresh}, "
          f"{N_REPLICATIONS} reps on {n_cores} cores", flush=True)
    rows = []
    with Pool(processes=n_cores, maxtasksperchild=1) as pool:
        for count, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            rows.append(row)
            if count % 10 == 0:
                print(f"  [{count}/{N_REPLICATIONS}] ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r[1])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rs.summary_columns(MAX_STEPS))
        writer.writerows(rows)
    print(f"  Saved -> {path}", flush=True)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    tid_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid_env is not None:
        idx = int(tid_env) - 1
        if not 0 <= idx < len(CONDITIONS):
            sys.exit(f"Task ID {tid_env} out of range for {len(CONDITIONS)} conditions.")
        indices = [idx]
    else:
        indices = list(range(len(CONDITIONS)))

    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except (ValueError, TypeError):
        slurm_cpus = 0
    n_cores = min(max(1, slurm_cpus or cpu_count() - 1), SAFE_CORE_LIMIT)
    print(f"Output: {OUT_DIR} ({N_REPLICATIONS} reps x {MAX_STEPS} steps, "
          f"root seed {ROOT_SEED}, lambda per condition {LAMBDA_VALUES}, victim_mode=none)")

    for idx in indices:
        run_condition(idx, n_cores)
    print("LAMBDA SWEEP JOB COMPLETE")


if __name__ == "__main__":
    main()
