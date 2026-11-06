"""Preflight check. Run this once on the cluster before submitting a sweep.

    python check_environment.py

Prints package versions, runs a small simulation, and checks that a seed
reproduces the same run twice.
"""
import os
import platform
import sys
import time


def main():
    print(f"python      {sys.version.split()[0]}  ({platform.platform()})")
    try:
        import numpy, scipy, mesa
    except ImportError as exc:
        sys.exit(f"MISSING PACKAGE: {exc}. Create the environment with "
                 "'conda env create -f environment.yml' (see CLUSTER.md).")
    print(f"numpy       {numpy.__version__}")
    print(f"scipy       {scipy.__version__}")
    print(f"mesa        {mesa.__version__}")
    if int(mesa.__version__.split(".")[0]) < 3:
        sys.exit("Mesa 3.x is required (the model uses mesa.discrete_space and shuffle_do).")
    for optional in ("pandas", "matplotlib", "statsmodels"):
        try:
            print(f"{optional:<12}{__import__(optional).__version__}")
        except ImportError:
            print(f"{optional:<12}not installed (needed only for the analysis scripts)")
    print(f"cpus visible to this process: {len(os.sched_getaffinity(0))}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import rel_DMAP_model
    import cognitive_functions as cf
    import numpy as np

    w = cf.prelec(0, 1, 0.8)
    print(f"w(.80) at beta = 0, alpha = 1: {w:.3f}  (must be 0.800)")
    assert abs(w - 0.8) < 1e-9, "weighting function is not the corrected version"

    def one_run(seed):
        np.random.seed(seed)
        m = rel_DMAP_model(width=30, height=30, income_rank_threshold=0.20, seed=seed)
        for _ in range(5):
            m.step()
        return [(a.unique_id, float(a.wealth), a.decision) for a in m.agents]

    t0 = time.time()
    a, b = one_run(11), one_run(11)
    print(f"30 x 30, 5 steps, twice: {time.time() - t0:.1f}s")
    if sorted(a) == sorted(b):
        print("seeding is reproducible")
    else:
        sys.exit("SAME SEED GAVE DIFFERENT RESULTS - do not submit the sweep")
    print("\nEnvironment looks good. Next: sbatch slurm/smoke_test.sh")


if __name__ == "__main__":
    main()
