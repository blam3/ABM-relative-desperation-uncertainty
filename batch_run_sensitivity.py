"""
Sensitivity analyses (revised 2026-09).

Task IDs  1- 9  Softmax robustness
    tau_softmax in {1, 3, 5} x theta_softmax in {0.5, 1.5, 3.0}
    Fixed at the no-distortion centre: beta = 0, alpha = 1, threshold = 0.35
    (the June version used beta = 1, which was the no-distortion value only under
    the legacy weighting parameterization)
    25 steps, 100 replications

Task IDs 10-18  Convergence check
    beta in {-1, 0, 1} x threshold in {0.05, 0.35, 0.65}; alpha = 1
    50 steps, 100 replications

Output goes to DMURD_RESULTS_DIR/sensitivity_softmax_v2 and
DMURD_RESULTS_DIR/sensitivity_convergence_v2 (default under results/). Model
options, seeding, and the raw/gzip switches come from batch_run_hpc.py, so the
same environment variables apply.

Usage: sbatch slurm/submit_sensitivity.sh   or   python batch_run_sensitivity.py
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
BASE = os.path.dirname(main_sweep.RUN_FOLDER) or "results"
SUFFIX = os.environ.get("DMURD_SENSITIVITY_SUFFIX", "_v2")
DIR_SOFTMAX = os.path.join(BASE, f"sensitivity_softmax{SUFFIX}")
DIR_CONV = os.path.join(BASE, f"sensitivity_convergence{SUFFIX}")

SOFTMAX_CONDITIONS = list(itertools.product([1.0, 3.0, 5.0], [0.5, 1.5, 3.0]))
SOFTMAX_FIXED = dict(beta_loc=0, alpha_loc=1, income_rank_threshold=0.35, max_steps=25)

CONV_CONDITIONS = list(itertools.product([-1, 0, 1], [0.05, 0.35, 0.65]))
CONV_FIXED = dict(alpha_loc=1, max_steps=50)


def seed_for(task_id, rep_id):
    return int(np.random.SeedSequence([ROOT_SEED, 2000 + task_id, rep_id]).generate_state(1)[0])


def worker(args):
    task_id, rep_id, beta, alpha, thresh, tau_sm, theta_sm, max_steps = args
    seed = seed_for(task_id, rep_id)
    np.random.seed(seed)
    opts = dict(MODEL_OPTIONS)
    opts.update(tau_softmax=tau_sm, theta_softmax=theta_sm)
    model = OptimizedModel(beta_loc=beta, alpha_loc=alpha, income_rank_threshold=thresh,
                           seed=seed, **opts)
    agents = list(model.agents)
    init_gap = rs.gap([a.wealth for a in agents])
    crime, desperate_share = [], []
    try:
        for _ in range(max_steps):
            model.step()
            c, s = rs.step_stats(agents)
            crime.append(c)
            desperate_share.append(s)
        meta = (task_id, rep_id, seed, beta, alpha, thresh, tau_sm, theta_sm,
                MODEL_OPTIONS["desperation_rule"])
        summary = rs.summarize(agents, crime, desperate_share, meta, init_gap)
    finally:
        del model
        gc.collect()
    return summary


def task_spec(tid):
    if 1 <= tid <= 9:
        tau_sm, theta_sm = SOFTMAX_CONDITIONS[tid - 1]
        f = SOFTMAX_FIXED
        path = os.path.join(DIR_SOFTMAX, f"summary_softmax_tau{tau_sm}_theta{theta_sm}.csv")
        return path, (f["beta_loc"], f["alpha_loc"], f["income_rank_threshold"],
                      tau_sm, theta_sm, f["max_steps"])
    if 10 <= tid <= 18:
        beta, thresh = CONV_CONDITIONS[tid - 10]
        safe_beta = str(beta).replace("-", "neg")
        path = os.path.join(DIR_CONV, f"summary_conv_beta{safe_beta}_thresh{thresh}_50steps.csv")
        return path, (beta, CONV_FIXED["alpha_loc"], thresh, MODEL_OPTIONS["tau_softmax"],
                      MODEL_OPTIONS["theta_softmax"], CONV_FIXED["max_steps"])
    return None, None


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(DIR_SOFTMAX, exist_ok=True)
    os.makedirs(DIR_CONV, exist_ok=True)
    tid_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_ids = [int(tid_env)] if tid_env else list(range(1, 19))
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except (ValueError, TypeError):
        slurm_cpus = 0
    n_cores = min(max(1, slurm_cpus or cpu_count() - 1), SAFE_CORE_LIMIT)

    for tid in task_ids:
        path, spec = task_spec(tid)
        if path is None:
            print(f"Task ID {tid} out of range [1-18]. Skipping.")
            continue
        max_steps = spec[-1]
        t0 = time.time()
        print(f"Task {tid}: beta={spec[0]}, alpha={spec[1]}, threshold={spec[2]}, "
              f"tau={spec[3]}, theta={spec[4]}, {max_steps} steps", flush=True)
        tasks = [(tid, r) + spec for r in range(N_REPLICATIONS)]
        rows = []
        with Pool(processes=n_cores, maxtasksperchild=1) as pool:
            for count, row in enumerate(pool.imap_unordered(worker, tasks), 1):
                rows.append(row)
                if count % 10 == 0:
                    print(f"  [{count}/{N_REPLICATIONS}] ({time.time() - t0:.0f}s)", flush=True)
        rows.sort(key=lambda r: r[1])
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rs.summary_columns(max_steps))
            writer.writerows(rows)
        print(f"  Saved -> {path}", flush=True)
    print("SENSITIVITY JOB COMPLETE")


if __name__ == "__main__":
    main()
