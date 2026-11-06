"""
Main parameter sweep for the DMURD model (revised 2026-09).

3 beta x 3 alpha x 5 relative-desperation thresholds x 100 replications = 4,500 runs.
Designed to run as a 45-task Slurm array job (one condition per task); see
slurm/submit_main_sweep.sh and CLUSTER.md.

Changes from the June 2026 version
1. Model options: weighting="prelec_log_elevation", alpha_dist="lognormal",
   alpha_scale=0.35 (see cognitive_functions.py and agents.py).
2. Seeding: the June version used seed = 12345 + rep + int(|beta|*100 + alpha*10 +
   thresh*1000), which produced only 815 distinct seeds for 4,500 runs and gave the
   beta = -1 and beta = +1 conditions identical random streams. Seeds now come from
   numpy.random.SeedSequence([ROOT_SEED, condition, rep]).
3. Every run writes a replication-level summary row (run_summary.py). The
   per-agent rows are optional and can be gzipped, because the full sweep writes
   about 110 GB uncompressed.

Environment variables (all optional)
    DMURD_RESULTS_DIR   output directory        default results/final_batch_run_v2
    DMURD_RAW           1 = per-agent rows      default 1
    DMURD_GZIP          1 = gzip per-agent rows default 0
    DMURD_REPLICATIONS  replications per cell   default 100
    DMURD_STEPS         steps per run           default 25
    DMURD_ROOT_SEED     root seed               default 20260915

Usage: sbatch slurm/submit_main_sweep.sh   or   python batch_run_hpc.py
"""
import sys
import os
import csv
import gzip
import gc
import itertools
import time
import numpy as np
from multiprocessing import Pool, cpu_count

from model import rel_DMAP_model
import cognitive_functions as cf
import run_summary as rs

# --- CONFIGURATION ---
GRID_WIDTH = GRID_HEIGHT = int(os.environ.get("DMURD_GRID", 100))  # DMURD_GRID shrinks the grid for smoke tests
MAX_STEPS = int(os.environ.get("DMURD_STEPS", 25))
N_REPLICATIONS = int(os.environ.get("DMURD_REPLICATIONS", 100))
ROOT_SEED = int(os.environ.get("DMURD_ROOT_SEED", 20260915))
SAFE_CORE_LIMIT = 24

RUN_FOLDER = os.environ.get("DMURD_RESULTS_DIR", os.path.join("results", "final_batch_run_v2"))
WRITE_RAW = os.environ.get("DMURD_RAW", "1") == "1"
GZIP_RAW = os.environ.get("DMURD_GZIP", "0") == "1"

MODEL_OPTIONS = dict(
    lambd=float(os.environ.get("DMURD_LAMBDA", 0.2)),
    gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
    beta_scale=0.5, alpha_scale=0.35,
    weighting="prelec_log_elevation", alpha_dist="lognormal",
    desperation_rule="rank",
    min_start_wealth=100, pareto_shape=3, pareto_scale=200,
    p=0.8, width=GRID_WIDTH, height=GRID_HEIGHT,
    num_neighbors=20, tau_softmax=3.0, theta_softmax=1.5,
)

# Parameter sweep values (beta = 0 is the no-distortion baseline)
BETA_LOC_VALUES = [-1, 0, 1]
ALPHA_LOC_VALUES = [0.5, 1, 2]
THRESHOLD_VALUES = [0.05, 0.2, 0.35, 0.5, 0.65]
ALL_COMBINATIONS = list(itertools.product(BETA_LOC_VALUES, ALPHA_LOC_VALUES, THRESHOLD_VALUES))


class OptimizedModel(rel_DMAP_model):
    """Skips the DataCollector to save memory; rows are extracted manually."""
    def step(self):
        self.agents.shuffle_do("step")


def raw_header():
    return ["ReplicationID", "Step", "AgentID", "Wealth", "WealthStart",
            "Decision", "Rank", "Desperate", "RelDeprivation", "Position", "Caught",
            "Beta", "Alpha", "Beta_Loc", "Alpha_Loc", "Threshold"]


def run_seed(condition_index, rep_id):
    return int(np.random.SeedSequence([ROOT_SEED, condition_index, rep_id]).generate_state(1)[0])


def run_simulation_worker(args):
    """One replication. Returns (summary_row, raw_rows)."""
    condition_index, rep_id, beta, alpha, thresh = args
    current_seed = run_seed(condition_index, rep_id)
    np.random.seed(current_seed)  # scipy/numpy draws inside agents use the global RNG

    model = OptimizedModel(beta_loc=beta, alpha_loc=alpha,
                           income_rank_threshold=thresh, seed=current_seed,
                           **MODEL_OPTIONS)
    agents = list(model.agents)
    init_gap = rs.gap([a.wealth for a in agents])
    mean_weight = float(np.mean([cf.prelec(a.beta, a.alpha, MODEL_OPTIONS["p"],
                                           MODEL_OPTIONS["weighting"]) for a in agents]))

    crime, desperate_share, raw_rows = [], [], []
    try:
        for step in range(MAX_STEPS):
            model.step()
            c, s = rs.step_stats(agents)
            crime.append(c)
            desperate_share.append(s)
            if WRITE_RAW:
                # values are rounded to keep the sweep's per-agent output near 40 GB
                raw_rows.extend(
                    [rep_id, step, a.unique_id, round(float(a.wealth), 2),
                     round(float(a.wealth_start), 2), a.decision, round(a.income_rank, 4),
                     int(a.desperate_state), round(float(a.rel_deprivation), 2),
                     f"{a.pos[0]};{a.pos[1]}", int(a.caught), round(float(a.beta), 4),
                     round(float(a.alpha), 4), beta, alpha, thresh]
                    for a in agents
                )
        meta = (condition_index, rep_id, current_seed, beta, alpha, thresh,
                MODEL_OPTIONS["tau_softmax"], MODEL_OPTIONS["theta_softmax"],
                MODEL_OPTIONS["desperation_rule"])
        summary = rs.summarize(agents, crime, desperate_share, meta, init_gap, mean_weight)
    finally:
        del model
        gc.collect()
    return summary, raw_rows


def open_raw(path):
    if GZIP_RAW:
        return gzip.open(path + ".gz", "wt", newline="")
    return open(path, "w", newline="")


def run_condition(idx, n_cores):
    os.makedirs(RUN_FOLDER, exist_ok=True)
    beta, alpha, thresh = ALL_COMBINATIONS[idx]
    cond = idx + 1
    safe_beta = str(beta).replace("-", "neg")
    stem = f"cond_{cond}_beta_{safe_beta}_alpha_{alpha}_thresh_{thresh}"
    raw_path = os.path.join(RUN_FOLDER, stem + ".csv")
    summary_path = os.path.join(RUN_FOLDER, "summary_" + stem + ".csv")
    tasks = [(cond, r, beta, alpha, thresh) for r in range(N_REPLICATIONS)]
    t0 = time.time()
    print(f"Condition {cond}: beta={beta}, alpha={alpha}, threshold={thresh}, "
          f"{N_REPLICATIONS} replications on {n_cores} cores", flush=True)

    summaries = []
    raw_file = open_raw(raw_path) if WRITE_RAW else None
    try:
        raw_writer = csv.writer(raw_file) if WRITE_RAW else None
        if WRITE_RAW:
            raw_writer.writerow(raw_header())
        with Pool(processes=n_cores, maxtasksperchild=1) as pool:
            for count, (summary, rows) in enumerate(
                    pool.imap_unordered(run_simulation_worker, tasks), 1):
                summaries.append(summary)
                if WRITE_RAW:
                    raw_writer.writerows(rows)
                if count % 10 == 0:
                    if WRITE_RAW:
                        raw_file.flush()
                    print(f"   [Cond {cond}] {count}/{N_REPLICATIONS} runs "
                          f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        if raw_file is not None:
            raw_file.close()

    summaries.sort(key=lambda r: r[1])
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rs.summary_columns(MAX_STEPS))
        writer.writerows(summaries)
    print(f"   -> {summary_path}" + (f" and {raw_path}{'.gz' if GZIP_RAW else ''}"
                                     if WRITE_RAW else " (summaries only)"), flush=True)
    gc.collect()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(RUN_FOLDER, exist_ok=True)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id) - 1
        if not 0 <= idx < len(ALL_COMBINATIONS):
            sys.exit(f"Task ID {task_id} is out of range for {len(ALL_COMBINATIONS)} conditions.")
        indices = [idx]
        print(f"SLURM ARRAY MODE: condition {task_id} of {len(ALL_COMBINATIONS)}")
    else:
        indices = list(range(len(ALL_COMBINATIONS)))
        print("STANDARD MODE: running all conditions sequentially.")

    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except (ValueError, TypeError):
        slurm_cpus = 0
    n_cores = min(slurm_cpus if slurm_cpus > 0 else max(1, cpu_count() - 1), SAFE_CORE_LIMIT)
    print(f"Output: {RUN_FOLDER} (raw={'gzip' if GZIP_RAW else WRITE_RAW}, "
          f"{N_REPLICATIONS} reps x {MAX_STEPS} steps, root seed {ROOT_SEED})")

    for idx in indices:
        run_condition(idx, n_cores)
    print("JOB COMPLETE")


if __name__ == "__main__":
    main()
