"""Record full-precision trajectories from one code tree (used by test_victimization).

Run in a subprocess with `python -I` so that only the tree given by --tree is
imported. Every DMURD_* variable is removed from the environment first, then the
small-grid settings below are set, so both trees read identical configuration.

    python -I _trajectory_driver.py --tree DIR --out FILE.pkl [--explicit-none]

--explicit-none passes victim_mode="none", sync=False to the model constructor
(the new tree only; jasss-v1 has no such arguments).
"""
import argparse
import os
import pickle
import sys

GRID, STEPS = 20, 25
CONDITIONS = [(0, 1, 0.35), (-1, 2, 0.65)]
REPS = [0, 1, 2]
AGENT_FIELDS = ["wealth", "wealth_start", "decision", "caught", "desperate_state",
                "income_rank", "rel_deprivation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--explicit-none", action="store_true")
    args = ap.parse_args()

    for k in [k for k in os.environ if k.startswith("DMURD_")]:
        del os.environ[k]
    os.environ.update(DMURD_GRID=str(GRID), DMURD_STEPS=str(STEPS), DMURD_RAW="1",
                      DMURD_REPLICATIONS=str(len(REPS)))
    tree = os.path.abspath(args.tree)
    sys.path.insert(0, tree)
    os.chdir(tree)

    import numpy as np
    import batch_run_hpc as bh
    from model import rel_DMAP_model

    extra = dict(victim_mode="none", sync=False) if args.explicit_none else {}
    out = {"tree": tree, "explicit_none": args.explicit_none, "runs": {}}
    for beta, alpha, nu in CONDITIONS:
        cond = bh.ALL_COMBINATIONS.index((beta, alpha, nu)) + 1
        for rep in REPS:
            seed = bh.run_seed(cond, rep)
            # 1. full-precision per-agent trajectories through the model's own step()
            np.random.seed(seed)
            model = rel_DMAP_model(beta_loc=beta, alpha_loc=alpha, income_rank_threshold=nu,
                                   seed=seed, **bh.MODEL_OPTIONS, **extra)
            agents = sorted(model.agents, key=lambda a: a.unique_id)
            traj = {f: [] for f in AGENT_FIELDS + ["pos", "beta", "alpha"]}
            for _ in range(STEPS):
                model.step()
                for f in AGENT_FIELDS:
                    traj[f].append(np.array([float(getattr(a, f)) for a in agents]))
                traj["pos"].append(np.array([a.pos for a in agents]))
            traj["beta"] = [np.array([float(a.beta) for a in agents])]
            traj["alpha"] = [np.array([float(a.alpha) for a in agents])]
            model_vars = model.datacollector.get_model_vars_dataframe()
            # 2. the batch pipeline (OptimizedModel, summary row, raw rows)
            summary, raw = bh.run_simulation_worker((cond, rep, beta, alpha, nu))
            out["runs"][(beta, alpha, nu, rep)] = {
                "seed": seed,
                "traj": {k: np.stack(v) for k, v in traj.items()},
                "model_vars": model_vars.to_numpy(),
                "model_var_cols": list(model_vars.columns),
                "summary": summary,
                "summary_cols": bh.rs.summary_columns(STEPS),
                "raw": raw,
                "raw_header": bh.raw_header(),
            }
    with open(args.out, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
