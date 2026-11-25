"""Run a small, seeded demonstration; use explicit sizes for research-scale runs."""
import argparse
import json
from pathlib import Path
import platform

import mesa
import numpy as np
import pandas as pd
import scipy

from model import rel_DMAP_model


def simulate(width=20, height=20, steps=10, seed=20260918):
    """Preserve model dynamics while seeding both legacy NumPy/SciPy and Mesa RNGs.

    Agent draws currently use NumPy's global RNG; a Mesa seed alone is not enough.
    Run separate simulations in separate processes for parallel research sweeps.
    """
    if width < 7 or height < 7:
        raise ValueError("width and height must be at least 7 for the radius-3 neighborhood")
    if steps < 1:
        raise ValueError("steps must be positive")
    np.random.seed(seed)
    model = rel_DMAP_model(
        lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
        beta_loc=0, beta_scale=0.5, alpha_loc=1, alpha_scale=0.35,
        weighting="prelec_log_elevation", alpha_dist="lognormal",
        min_start_wealth=100, pareto_shape=3, pareto_scale=200, p=0.8,
        width=width, height=height, num_neighbors=20, income_rank_threshold=0.20,
        tau_softmax=3.0, theta_softmax=1.5, seed=seed,
    )
    for _ in range(steps):
        model.step()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--agent-data", action="store_true", help="also export per-agent rows")
    parser.add_argument("--plot", action="store_true", help="write a demonstration trajectory figure")
    args = parser.parse_args()
    if args.width < 7 or args.height < 7 or args.steps < 1:
        parser.error("width/height must be >=7 and steps must be >=1")
    model = simulate(args.width, args.height, args.steps, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = model.datacollector.get_model_vars_dataframe()
    frame.index.name = "Step"
    frame.to_csv(args.out / "model_data.csv")
    if args.agent_data:
        model.datacollector.get_agent_vars_dataframe().to_csv(args.out / "agent_data.csv")
    metadata = {
        "purpose": "single-seed software demonstration; not research evidence",
        "width": args.width, "height": args.height, "steps": args.steps, "seed": args.seed,
        "initial_row": "Step 0 is initialization; decisions have not yet been made",
        "parameters": {"lambda": 0.2, "gamma": 0.3, "p_success": 0.8,
                       "beta_loc": 0, "beta_scale": 0.5, "alpha_loc": 1,
                       "alpha_scale": 0.35, "income_rank_threshold": 0.20,
                       "tau_softmax": 3.0, "theta_softmax": 1.5},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__, "mesa": mesa.__version__},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        # Exclude initialization because its decisions are all zero by construction.
        frame.loc[1:, "Proportion crime"].plot(ax=axes[0], color="#206b61", marker="o")
        axes[0].set(ylabel="Proportion choosing to offend", ylim=(0, 1), xlabel="Model step")
        frame.loc[1:, "P90-P10"].plot(ax=axes[1], color="#875c2c", marker="o")
        axes[1].set(ylabel="P90 − P10 wealth gap (model units)", xlabel="Model step")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Software demonstration · {args.width * args.height} agents · seed {args.seed}\nSingle run; not a calibrated crime forecast", fontsize=12)
        fig.savefig(args.out / "trajectories.png", dpi=160)
        plt.close(fig)
    print(f"Wrote {len(frame)} model rows (including initialization) to {args.out}")
    print("Single-seed demonstration only; use replicated sweeps for research conclusions.")


if __name__ == "__main__":
    main()
