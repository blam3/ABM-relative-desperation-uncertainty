"""Replication-level summary of one simulation run.

The regression metamodel in the manuscript treats one replication as the unit of
analysis, so every batch script writes these rows. They are small (one row per
run) and are what the analysis reads; the per-agent rows are only needed for
agent-level analyses.
"""
import numpy as np


def summary_columns(max_steps):
    return (["Condition", "ReplicationID", "Seed", "Beta_Loc", "Alpha_Loc", "Threshold",
             "Tau_Softmax", "Theta_Softmax", "Rule", "InitGap"]
            + [f"Crime_{s + 1}" for s in range(max_steps)]
            + [f"Desperate_Share_{s + 1}" for s in range(max_steps)]
            + ["Crime_Final", "Desperate_Rate_Final", "NonDesperate_Rate_Final",
               "Desperate_Share_Final", "Mean_Wealth", "Median_Wealth", "P10_Wealth",
               "P90_Wealth", "Gap_Final", "Frac_Debt", "Caught_Final",
               "LPM_b_Desperate", "LPM_b_Wealth", "Mean_Weight_p", "Mean_Beta", "Mean_Alpha"])


def gap(w):
    return float(np.percentile(w, 90) - np.percentile(w, 10))


def step_stats(agents):
    """Called after each model step; returns (crime proportion, desperate share)."""
    d = np.fromiter((a.decision for a in agents), dtype=float, count=len(agents))
    s = np.fromiter((a.desperate_state for a in agents), dtype=float, count=len(agents))
    return float(d.mean()), float(s.mean())


def summarize(agents, crime, desperate_share, meta, init_gap, mean_weight=np.nan):
    """Build one summary row. `crime` and `desperate_share` are per-step lists."""
    d = np.fromiter((a.decision for a in agents), dtype=float, count=len(agents))
    s = np.fromiter((a.desperate_state for a in agents), dtype=float, count=len(agents))
    w = np.fromiter((a.wealth for a in agents), dtype=float, count=len(agents))
    w0 = np.fromiter((a.wealth_start for a in agents), dtype=float, count=len(agents))
    caught = np.fromiter((float(a.caught) for a in agents), dtype=float, count=len(agents))
    sd = w0.std()
    if sd > 0:
        X = np.column_stack([np.ones(len(d)), s, (w0 - w0.mean()) / sd])
        coef = np.linalg.lstsq(X, d, rcond=None)[0]
        b_desp, b_wealth = float(coef[1]), float(coef[2])
    else:
        b_desp = b_wealth = float("nan")
    return (list(meta) + [init_gap] + [float(c) for c in crime] + [float(x) for x in desperate_share]
            + [float(d.mean()),
               float(d[s == 1].mean()) if s.sum() else float("nan"),
               float(d[s == 0].mean()) if (s == 0).sum() else float("nan"),
               float(s.mean()), float(w.mean()), float(np.median(w)),
               float(np.percentile(w, 10)), float(np.percentile(w, 90)), gap(w),
               float((w < 0).mean()), float(caught.mean()), b_desp, b_wealth,
               float(mean_weight),
               float(np.mean([a.beta for a in agents])),
               float(np.mean([a.alpha for a in agents]))])
