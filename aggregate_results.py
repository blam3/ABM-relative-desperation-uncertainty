"""Collect the replication-level summaries into one analysis file.

    python aggregate_results.py [--results DIR] [--out DIR] [--metamodel]

Reads every summary_*.csv under the results directory (default the main sweep),
writes analysis/replication_level.csv (one row per run) and
analysis/condition_means.csv, and with --metamodel fits the OLS metamodel
reported in the manuscript: offending at the final step on standardized beta,
likelihood insensitivity (iota = -log2 alpha), and the threshold, with eta
squared (which sums to R squared) alongside partial eta squared.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_RESULTS = os.environ.get("DMURD_RESULTS_DIR", os.path.join("results", "final_batch_run_v2"))


def load(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "summary_*.csv")))
    if not files:
        sys.exit(f"No summary_*.csv files in {results_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"{len(files)} condition files, {len(df)} runs")
    return df


def condition_means(df):
    keys = ["Beta_Loc", "Alpha_Loc", "Threshold"]
    cols = [c for c in ("Crime_Final", "Desperate_Rate_Final", "NonDesperate_Rate_Final",
                        "Desperate_Share_Final", "Gap_Final", "Median_Wealth", "Frac_Debt",
                        "LPM_b_Desperate", "LPM_b_Wealth") if c in df]
    out = df.groupby(keys)[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["Desperation_Gap"] = (df.groupby(keys).Desperate_Rate_Final.mean()
                              - df.groupby(keys).NonDesperate_Rate_Final.mean())
    return out.reset_index()


def metamodel(df):
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        print("statsmodels is not installed; skipping the metamodel")
        return None
    x = df.copy()
    x["iota"] = -np.log2(x.Alpha_Loc)                      # higher = more insensitive
    for src, dst in (("Beta_Loc", "zb"), ("iota", "zi"), ("Threshold", "zt")):
        x[dst] = (x[src] - x[src].mean()) / x[src].std()
    fit = smf.ols("Crime_Final ~ zb * zi * zt", data=x).fit()
    a = anova_lm(fit, typ=3)
    ssr = a.loc["Residual", "sum_sq"]
    sst = ((x.Crime_Final - x.Crime_Final.mean()) ** 2).sum()
    tab = pd.DataFrame({
        "b": fit.params,
        "eta2": a["sum_sq"] / sst,
        "partial_eta2": a["sum_sq"] / (a["sum_sq"] + ssr),
    }).drop(["Intercept", "Residual"], errors="ignore")
    print(f"\nOLS metamodel (unit = one replication), R2 = {fit.rsquared:.3f}")
    print("zb = uncertainty aversion, zi = likelihood insensitivity (-log2 alpha), "
          "zt = desperation threshold")
    print(tab.round(3).to_string())
    print(f"eta squared sums to {tab.eta2.sum():.3f} (= R2); partial eta squared does not sum.")
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--metamodel", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.results)
    rep_path = os.path.join(args.out, "replication_level.csv")
    df.to_csv(rep_path, index=False)
    cm = condition_means(df)
    cond_path = os.path.join(args.out, "condition_means.csv")
    cm.to_csv(cond_path, index=False)
    print(f"wrote {rep_path} and {cond_path}")
    if args.metamodel:
        tab = metamodel(df)
        if tab is not None:
            tab.to_csv(os.path.join(args.out, "metamodel_terms.csv"))
