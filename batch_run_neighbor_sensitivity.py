"""
Comparison group size (n) sensitivity analysis (Supplementary Text 2).

Design
------
n in {5, 10, 20, 30, 48} crossed with beta in {-1, 0, +1} and nu in
{0.05, 0.20, 0.35, 0.50, 0.65}. alpha = 1.0 (no likelihood distortion).
75 cells x 20 replications = 1,500 runs. 15 x 15 grid, 25 steps.

Output: BASE/neighbor_sensitivity_<suffix>/summary_n<n>_beta<b>_nu<t>.csv
        one row per replication, no per-agent output.

Usage:  sbatch slurm/submit_neighbor_sensitivity.sh   or
        python batch_run_neighbor_sensitivity.py
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

N_REPLICATIONS = int(os.environ.get("DMURD_NEIGHBOR_REPS", 20))
MAX_STEPS = 25
GRID = int(os.environ.get("DMURD_NEIGHBOR_GRID", 15))
BASE = os.path.dirname(main_sweep.RUN_FOLDER) or "results"
SUFFIX = os.environ.get("DMURD_SENSITIVITY_SUFFIX", "_v2")
OUT_DIR = os.path.join(BASE, f"neighbor_sensitivity{SUFFIX}")

N_VALUES = [5, 10, 20, 30, 48]
BETA_VALUES = [-1, 0, 1]
NU_VALUES = [0.05, 0.20, 0.35, 0.50, 0.65]
CONDITIONS = list(itertools.product(N_VALUES, BETA_VALUES, NU_VALUES))

FIXED_ALPHA = 1


def seed_for(task_id, rep_id):
    # 4000 offset keeps this sweep's seed stream disjoint from other sweeps.
    return int(np.random.SeedSequence([ROOT_SEED, 4000 + task_id, rep_id]).generate_state(1)[0])


def worker(args):
    task_id, rep_id, n, beta, nu = args
    seed = seed_for(task_id, rep_id)
    np.random.seed(seed)
    opts = dict(MODEL_OPTIONS)
    opts.update(num_neighbors=n, width=GRID, height=GRID)
    model = OptimizedModel(beta_loc=beta, alpha_loc=FIXED_ALPHA,
                           income_rank_threshold=nu, seed=seed, **opts)
    agents = list(model.agents)
    init_gap = rs.gap([a.wealth for a in agents])
    crime, desperate_share = [], []
    try:
        for _ in range(MAX_STEPS):
            model.step()
            c, s = rs.step_stats(agents)
            crime.append(c)
            desperate_share.append(s)
        meta = (task_id, rep_id, seed, beta, FIXED_ALPHA, nu,
                MODEL_OPTIONS["tau_softmax"], MODEL_OPTIONS["theta_softmax"],
                MODEL_OPTIONS["desperation_rule"])
        summary = rs.summarize(agents, crime, desperate_share, meta, init_gap)
    finally:
        del model
        gc.collect()
    return summary


def run_condition(idx, n_cores):
    n, beta, nu = CONDITIONS[idx]
    tid = idx + 1
    safe_beta = str(beta).replace("-", "neg")
    path = os.path.join(OUT_DIR, f"summary_n{n}_beta{safe_beta}_nu{nu}.csv")
    tasks = [(tid, r, n, beta, nu) for r in range(N_REPLICATIONS)]
    t0 = time.time()
    print(f"Task {tid}/{len(CONDITIONS)}: n={n}, beta={beta}, nu={nu}, "
          f"{N_REPLICATIONS} reps on {n_cores} cores, {GRID}x{GRID} grid", flush=True)
    rows = []
    with Pool(processes=n_cores, maxtasksperchild=1) as pool:
        for count, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            rows.append(row)
    rows.sort(key=lambda r: r[1])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rs.summary_columns(MAX_STEPS))
        writer.writerows(rows)
    print(f"  Saved -> {path} ({time.time() - t0:.0f}s)", flush=True)


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
          f"{GRID}x{GRID} grid, root seed {ROOT_SEED}, lambda={MODEL_OPTIONS['lambd']})")

    for idx in indices:
        run_condition(idx, n_cores)
    print("NEIGHBOR SENSITIVITY JOB COMPLETE")


if __name__ == "__main__":
    main()
