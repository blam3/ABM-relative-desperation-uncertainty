import sys
import os
import csv
import gc
import itertools
import time
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Import your original model
from model import rel_DMAP_model

# --- CONFIGURATION ---
GRID_WIDTH = 100
GRID_HEIGHT = 100
MAX_STEPS = 25
N_REPLICATIONS = 100

# Use fewer cores per node to prevent Memory Crash (OOM)
SAFE_CORE_LIMIT = 24 

# Parameter Sweep Values
BETA_LOC_VALUES = [-1, 0, 1]
ALPHA_LOC_VALUES = [0.5, 1, 2]
THRESHOLD_VALUES = [0.05, 0.2, 0.35, 0.5, 0.65]

# Output setup
BASE_OUTPUT_FOLDER = "results"
# We remove the timestamp from the folder name so all Array Jobs write to the same place
RUN_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "final_batch_run")

class OptimizedModel(rel_DMAP_model):
    """
    Overrides the step() function to:
    1. Disable the internal DataCollector (saves RAM).
    2. Use the modern Mesa agent scheduler.
    """
    def step(self):
        self.agents.shuffle_do("step")

def get_header():
    return [
        "ReplicationID", "Step", "AgentID", 
        "Wealth", "Decision", "Rank", "Desperate", 
        "Position", "Caught",
        "Beta_Loc", "Alpha_Loc", "Threshold"
    ]

def run_simulation_worker(args):
    """
    Worker function. Runs one simulation, extracts data manually.
    """
    rep_id, beta, alpha, thresh = args
    
    # 1. Deterministic Seeding
    seed_base = int(abs(beta)*100 + alpha*10 + thresh*1000)
    current_seed = 12345 + rep_id + seed_base
    
    # 2. Initialize Model
    model = OptimizedModel(
        lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250, 
        beta_loc=beta, alpha_loc=alpha, income_rank_threshold=thresh,
        beta_scale=0.5, alpha_scale=0.5, 
        min_start_wealth=100, p=0.8, width=GRID_WIDTH, height=GRID_HEIGHT, 
        num_neighbors=20,
        seed=current_seed
    )
    
    # 3. Manual Data Extraction
    all_rows = []
    
    try:
        for step in range(MAX_STEPS):
            model.step() 
            
            step_rows = [
                [
                    rep_id, step, agent.unique_id,
                    agent.wealth, agent.decision, agent.income_rank, 
                    int(agent.desperate_state), 
                    str(agent.pos), 
                    int(agent.caught),
                    beta, alpha, thresh
                ]
                for agent in model.agents
            ]
            all_rows.extend(step_rows)
            
    finally:
        del model
        gc.collect() 
    
    return all_rows

if __name__ == '__main__':
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(RUN_FOLDER, exist_ok=True)
    
    # 1. Generate ALL 45 Combinations
    all_combinations = list(itertools.product(
        BETA_LOC_VALUES, ALPHA_LOC_VALUES, THRESHOLD_VALUES
    ))
    
    # --- 2. ARRAY JOB LOGIC ---
    # Check if we are running inside a SLURM Array Job
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')

    if task_id is not None:
        # We are in an Array Job!
        # SLURM indexes usually start at 1, Python starts at 0.
        idx = int(task_id) - 1
        
        if 0 <= idx < len(all_combinations):
            # Pick ONLY the combination for this specific job
            my_combinations = [all_combinations[idx]]
            print(f"SLURM ARRAY MODE: Task ID {task_id}")
            print(f"Processing ONLY Condition {task_id} of {len(all_combinations)}")
        else:
            print(f"Error: Task ID {task_id} is out of range for {len(all_combinations)} conditions.")
            sys.exit(1)
    else:
        # Not in an array job (running locally or single testing)
        print("STANDARD MODE: Running ALL conditions sequentially.")
        my_combinations = all_combinations

    # --- 3. CORE DETECTION ---
    try:
        slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    except (ValueError, TypeError):
        slurm_cpus = 0

    if slurm_cpus > 0:
        n_cores = min(slurm_cpus, SAFE_CORE_LIMIT)
    else:
        n_cores = min(max(1, cpu_count() - 1), SAFE_CORE_LIMIT)
    
    print("="*60)
    print(f"Running on {n_cores} cores.")
    print("="*60)
    
    # --- 4. RUN LOOP ---
    for i, (beta, alpha, thresh) in enumerate(my_combinations):
        
        # NOTE: If in Array mode, 'i' is 0, but we want the REAL index for file naming
        # We find the real index from the master list
        real_index = all_combinations.index((beta, alpha, thresh)) + 1
        
        print(f"Starting Condition {real_index}: Beta={beta}, Alpha={alpha}, Thresh={thresh}...", flush=True)
        
        safe_beta = str(beta).replace("-", "neg")
        filename = f"cond_{real_index}_beta_{safe_beta}_alpha_{alpha}_thresh_{thresh}.csv"
        filepath = os.path.join(RUN_FOLDER, filename)
        
        tasks = [(r, beta, alpha, thresh) for r in range(N_REPLICATIONS)]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(get_header())
            f.flush()
            
            with Pool(processes=n_cores, maxtasksperchild=1) as pool:
                iterator = pool.imap_unordered(run_simulation_worker, tasks)
                
                count = 0
                for rows in iterator:
                    writer.writerows(rows)
                    count += 1
                    if count % 10 == 0:
                        f.flush()
                        print(f"   [Cond {real_index}] Processed {count}/{N_REPLICATIONS} runs...", flush=True)

        print(f"   -> Finished Condition {real_index}. File: {filename}\n", flush=True)
        gc.collect()

    print("JOB COMPLETE")
