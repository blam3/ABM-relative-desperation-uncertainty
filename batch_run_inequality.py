"""
Supplementary sweep for Goal 3 / H4 (added 2026-09).

Question: does inequality raise offending when inequality is set by the design
rather than produced by offending? Initial wealth is drawn as a mean-preserving
spread of the same Lomax shape (mean starting wealth is about 200 in every cell):
    low  : minimum 175, scale  50  (initial Gini about .08)
    mid  : minimum 100, scale 200  (initial Gini about .30; main-sweep value)
    high : minimum  25, scale 350  (initial Gini about .52)
Each spread is crossed with two desperation rules:
    rank         : income rank <= threshold (main model; threshold 0.20 and 0.50)
    median_ratio : wealth below 0.5 x median neighbor wealth (the share of
                   desperate agents can now grow with inequality)
beta and alpha are fixed at their no-distortion values (beta = 0, alpha = 1).
Usage: sbatch slurm/submit_inequality.sh   or   python batch_run_inequality.py
(writes <results>/inequality_sweep/summary.csv; honours DMURD_RESULTS_DIR,
DMURD_REPLICATIONS, DMURD_STEPS and DMURD_GRID)
"""
import os
import csv
import itertools
import numpy as np
from multiprocessing import Pool, cpu_count
import batch_run_hpc as main_sweep
from batch_run_hpc import OptimizedModel, MODEL_OPTIONS, ROOT_SEED

N_REPLICATIONS = main_sweep.N_REPLICATIONS
MAX_STEPS = main_sweep.MAX_STEPS
SPREADS = {"low": (175, 50), "mid": (100, 200), "high": (25, 350)}
RULES = [("rank", 0.20), ("rank", 0.50), ("median_ratio", None)]
OUT = os.path.join(os.path.dirname(main_sweep.RUN_FOLDER) or "results", "inequality_sweep")


def gap(w):
    return float(np.percentile(w, 90) - np.percentile(w, 10))


def worker(args):
    cell, rep, spread, rule, thresh = args
    seed = int(np.random.SeedSequence([ROOT_SEED, 1000 + cell, rep]).generate_state(1)[0])
    np.random.seed(seed)
    opts = dict(MODEL_OPTIONS)
    opts.update(min_start_wealth=SPREADS[spread][0], pareto_scale=SPREADS[spread][1],
                desperation_rule=rule, kappa=0.5)
    m = OptimizedModel(beta_loc=0, alpha_loc=1, income_rank_threshold=thresh or 0.0,
                       seed=seed, **opts)
    agents = list(m.agents)
    row = dict(rep=rep, seed=seed, spread=spread, rule=rule, thresh=thresh,
               init_gap=gap([a.wealth for a in agents]))
    for s in range(MAX_STEPS):
        m.step()
        d = np.array([a.decision for a in agents])
        ds = np.array([a.desperate_state for a in agents])
        row[f"crime_{s + 1}"] = d.mean()
        row[f"desperate_{s + 1}"] = ds.mean()
        row[f"gap_{s + 1}"] = gap([a.wealth for a in agents])
    return row


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cells = list(itertools.product(SPREADS, RULES))
    tasks = [(i, r, sp, ru, th) for i, (sp, (ru, th)) in enumerate(cells) for r in range(N_REPLICATIONS)]
    with Pool(max(1, min(cpu_count() - 1, 24)), maxtasksperchild=1) as pool, open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        writer = None
        for row in pool.imap_unordered(worker, tasks):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    print("DONE")
