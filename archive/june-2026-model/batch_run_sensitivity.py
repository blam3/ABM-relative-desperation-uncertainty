"""
Sensitivity analyses addressing three reviewer concerns.

Task IDs  1- 9  - Softmax robustness
    tau_softmax in {1, 3, 5} x theta_softmax in {0.5, 1.5, 3.0}  (9 combinations)
    Main params fixed at centre: beta=1, alpha=1, tau_income=0.35
    25 steps, 100 replications
    Output: results/sensitivity_softmax/

Task IDs 10-18  - Convergence check (50-step extended run)
    beta in {-1, 0, 1} x tau_income in {0.05, 0.35, 0.65}  (9 combinations)
    Fixed: alpha=1, softmax defaults (tau=3, theta=1.5)
    50 steps, 100 replications
    Output: results/sensitivity_convergence/

Usage
-----
    # SLURM array job (submit via submit_sensitivity_job.sh):
    sbatch submit_sensitivity_job.sh

    # Local test (all 18 conditions, sequential):
    python batch_run_sensitivity.py
"""
import sys
import os
import csv
import gc
import itertools
import numpy as np
from multiprocessing import Pool, cpu_count
from model import rel_DMAP_model

# Grid / replication config (same as main run)
GRID_WIDTH      = 100
GRID_HEIGHT     = 100
N_REPLICATIONS  = 100
SAFE_CORE_LIMIT = 24

# Output directories
BASE_OUTPUT = "results"
DIR_SOFTMAX = os.path.join(BASE_OUTPUT, "sensitivity_softmax")
DIR_CONV    = os.path.join(BASE_OUTPUT, "sensitivity_convergence")

# Softmax sweep - task IDs 1-9
TAU_SOFTMAX_VALUES   = [1.0, 3.0, 5.0]
THETA_SOFTMAX_VALUES = [0.5, 1.5, 3.0]
SOFTMAX_CONDITIONS   = list(itertools.product(TAU_SOFTMAX_VALUES, THETA_SOFTMAX_VALUES))
# Centre values for non-swept main parameters
SOFTMAX_FIXED = dict(beta_loc=1, alpha_loc=1, income_rank_threshold=0.35, max_steps=25)

# Convergence check - task IDs 10-18
CONV_BETA_VALUES   = [-1, 0, 1]
CONV_THRESH_VALUES = [0.05, 0.35, 0.65]
CONV_CONDITIONS    = list(itertools.product(CONV_BETA_VALUES, CONV_THRESH_VALUES))
CONV_FIXED = dict(alpha_loc=1, tau_softmax=3.0, theta_softmax=1.5, max_steps=50)


# Shared model subclass (no DataCollector, shuffle activation)
class OptimizedModel(rel_DMAP_model):
    def step(self):
        self.agents.shuffle_do("step")


# Headers
_BASE_HEADER = [
    "ReplicationID", "Step", "AgentID",
    "Wealth", "Decision", "Rank", "Desperate",
    "Position", "Caught",
    "Beta_Loc", "Alpha_Loc", "Threshold",
]
SOFTMAX_HEADER = _BASE_HEADER + ["Tau_Softmax", "Theta_Softmax"]
CONV_HEADER    = _BASE_HEADER  # convergence runs vary beta/thresh, same columns as main


# Worker: softmax sweep
def run_softmax_worker(args):
    rep_id, tau_sm, theta_sm = args
    seed_base    = int(tau_sm * 100 + theta_sm * 10)
    current_seed = 12345 + rep_id + seed_base
    np.random.seed(current_seed)

    model = OptimizedModel(
        lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
        beta_loc=SOFTMAX_FIXED["beta_loc"],
        alpha_loc=SOFTMAX_FIXED["alpha_loc"],
        income_rank_threshold=SOFTMAX_FIXED["income_rank_threshold"],
        beta_scale=0.5, alpha_scale=0.5,
        min_start_wealth=100, p=0.8,
        width=GRID_WIDTH, height=GRID_HEIGHT, num_neighbors=20,
        tau_softmax=tau_sm, theta_softmax=theta_sm,
        seed=current_seed,
    )
    all_rows = []
    try:
        for step in range(SOFTMAX_FIXED["max_steps"]):
            model.step()
            all_rows.extend([
                [rep_id, step, agent.unique_id,
                 agent.wealth, agent.decision, agent.income_rank,
                 int(agent.desperate_state), str(agent.pos), int(agent.caught),
                 SOFTMAX_FIXED["beta_loc"], SOFTMAX_FIXED["alpha_loc"],
                 SOFTMAX_FIXED["income_rank_threshold"],
                 tau_sm, theta_sm]
                for agent in model.agents
            ])
    finally:
        del model
        gc.collect()
    return all_rows


# Worker: convergence check
def run_convergence_worker(args):
    rep_id, beta, thresh = args
    seed_base    = int(abs(beta) * 100 + thresh * 1000)
    current_seed = 12345 + rep_id + seed_base
    np.random.seed(current_seed)

    model = OptimizedModel(
        lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
        beta_loc=beta,
        alpha_loc=CONV_FIXED["alpha_loc"],
        income_rank_threshold=thresh,
        beta_scale=0.5, alpha_scale=0.5,
        min_start_wealth=100, p=0.8,
        width=GRID_WIDTH, height=GRID_HEIGHT, num_neighbors=20,
        tau_softmax=CONV_FIXED["tau_softmax"],
        theta_softmax=CONV_FIXED["theta_softmax"],
        seed=current_seed,
    )
    all_rows = []
    try:
        for step in range(CONV_FIXED["max_steps"]):
            model.step()
            all_rows.extend([
                [rep_id, step, agent.unique_id,
                 agent.wealth, agent.decision, agent.income_rank,
                 int(agent.desperate_state), str(agent.pos), int(agent.caught),
                 beta, CONV_FIXED["alpha_loc"], thresh]
                for agent in model.agents
            ])
    finally:
        del model
        gc.collect()
    return all_rows


# Main
if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(DIR_SOFTMAX, exist_ok=True)
    os.makedirs(DIR_CONV,    exist_ok=True)

    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print("STANDARD MODE: running all 18 sensitivity conditions sequentially.")
        task_ids = list(range(1, 19))
    else:
        task_ids = [int(task_id_str)]
        print(f"SLURM ARRAY MODE: task {task_id_str}")

    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except (ValueError, TypeError):
        slurm_cpus = 0
    n_cores = min(max(1, slurm_cpus or cpu_count() - 1), SAFE_CORE_LIMIT)
    print(f"Running on {n_cores} cores.\n{'='*60}")

    for tid in task_ids:

        # Softmax sensitivity (tasks 1-9)
        if 1 <= tid <= 9:
            tau_sm, theta_sm = SOFTMAX_CONDITIONS[tid - 1]
            filename = f"softmax_tau{tau_sm}_theta{theta_sm}.csv"
            filepath = os.path.join(DIR_SOFTMAX, filename)
            print(f"Task {tid}: softmax sweep - tau_softmax={tau_sm}, theta_softmax={theta_sm}", flush=True)

            tasks = [(r, tau_sm, theta_sm) for r in range(N_REPLICATIONS)]
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(SOFTMAX_HEADER)
                with Pool(processes=n_cores, maxtasksperchild=1) as pool:
                    for count, rows in enumerate(
                            pool.imap_unordered(run_softmax_worker, tasks), 1):
                        writer.writerows(rows)
                        if count % 10 == 0:
                            f.flush()
                            print(f"  [{count}/{N_REPLICATIONS}] done", flush=True)
            print(f"  Saved -> {filepath}\n", flush=True)

        # Convergence check (tasks 10-18)
        elif 10 <= tid <= 18:
            beta, thresh = CONV_CONDITIONS[tid - 10]
            safe_beta = str(beta).replace("-", "neg")
            filename  = f"conv_beta{safe_beta}_thresh{thresh}_50steps.csv"
            filepath  = os.path.join(DIR_CONV, filename)
            print(f"Task {tid}: convergence - beta={beta}, tau_income={thresh}, 50 steps", flush=True)

            tasks = [(r, beta, thresh) for r in range(N_REPLICATIONS)]
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CONV_HEADER)
                with Pool(processes=n_cores, maxtasksperchild=1) as pool:
                    for count, rows in enumerate(
                            pool.imap_unordered(run_convergence_worker, tasks), 1):
                        writer.writerows(rows)
                        if count % 10 == 0:
                            f.flush()
                            print(f"  [{count}/{N_REPLICATIONS}] done", flush=True)
            print(f"  Saved -> {filepath}\n", flush=True)

        else:
            print(f"Task ID {tid} out of range [1-18]. Skipping.")

    print("SENSITIVITY JOB COMPLETE")
