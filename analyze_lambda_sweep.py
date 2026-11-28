"""
Inspect the lambda-sweep results (batch_run_lambda_sweep.py).

Reads every summary_lambda*_thresh*.csv from the lambda_sweep directory,
computes per-condition means and 95% bootstrap CIs for the desperate and
non-desperate final offending rates, checks monotonicity in nu for each
lambda, and writes a diagnostic plot.

Usage:
    python analyze_lambda_sweep.py [--dir /path/to/lambda_sweep]
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DIR = os.path.expanduser("~/scratch/dmurd/lambda_sweep")


def ci95(vals):
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = rng.choice(vals, size=(2000, n), replace=True).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def is_monotone_increasing(xs, ys, tol=0.0):
    """True if ys is (weakly) monotone increasing in the order of xs. tol lets
    tiny reversals slip through (per-step drop <= tol)."""
    order = np.argsort(xs)
    y_sorted = np.asarray(ys)[order]
    diffs = np.diff(y_sorted)
    return bool(np.all(diffs >= -tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None,
                    help="output png path (default: <dir>/lambda_sweep_diagnostic.png)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "summary_lambda*_thresh*.csv")))
    if not files:
        raise SystemExit(f"No summary files in {args.dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # The summary schema stores lambda via the tau_softmax column? No -- it stores
    # only the run_summary meta fields. Use the filename as ground truth.
    file_meta = []
    for f in files:
        base = os.path.basename(f)
        # summary_lambda{lam}_thresh{th}.csv
        stem = base.replace("summary_lambda", "").replace(".csv", "")
        lam_str, th_str = stem.split("_thresh")
        file_meta.append((float(lam_str), float(th_str), f))

    rows = []
    for lam, th, path in file_meta:
        sub = pd.read_csv(path)
        desp = sub["Desperate_Rate_Final"].values
        nond = sub["NonDesperate_Rate_Final"].values
        crime = sub["Crime_Final"].values
        d_lo, d_hi = ci95(desp)
        n_lo, n_hi = ci95(nond)
        c_lo, c_hi = ci95(crime)
        rows.append(dict(
            lambd=lam, threshold=th, n_reps=len(sub),
            desperate_mean=float(desp.mean()), desperate_lo=d_lo, desperate_hi=d_hi,
            nondesperate_mean=float(nond.mean()), nondesperate_lo=n_lo, nondesperate_hi=n_hi,
            crime_mean=float(crime.mean()), crime_lo=c_lo, crime_hi=c_hi,
        ))
    tbl = pd.DataFrame(rows).sort_values(["lambd", "threshold"]).reset_index(drop=True)

    print("\n=== Per-condition means ===")
    with pd.option_context("display.max_rows", None, "display.width", 140,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(tbl[["lambd", "threshold", "n_reps",
                   "desperate_mean", "nondesperate_mean", "crime_mean"]])

    print("\n=== Monotonicity of desperate offending in nu ===")
    print(f"{'lambda':>7}  {'desp @nu=0.05':>13}  {'@0.20':>7}  {'@0.35':>7}  "
          f"{'@0.50':>7}  {'@0.65':>7}  {'monotone?':>10}  {'delta_min-max':>14}")
    for lam in sorted(tbl.lambd.unique()):
        sub = tbl[tbl.lambd == lam].sort_values("threshold")
        thr = sub.threshold.values
        y = sub.desperate_mean.values
        mono_strict = is_monotone_increasing(thr, y, tol=0.0)
        mono_loose = is_monotone_increasing(thr, y, tol=0.005)
        rng = y.max() - y.min()
        label = "yes" if mono_strict else ("yes*" if mono_loose else "NO")
        print(f"{lam:>7.2f}  {y[0]:>13.4f}  {y[1]:>7.4f}  {y[2]:>7.4f}  "
              f"{y[3]:>7.4f}  {y[4]:>7.4f}  {label:>10}  {rng:>14.4f}")
    print("* = monotone within 0.005 tolerance")

    # Plot
    out_png = args.out or os.path.join(args.dir, "lambda_sweep_diagnostic.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    cmap = plt.cm.viridis
    lams = sorted(tbl.lambd.unique())
    for i, lam in enumerate(lams):
        sub = tbl[tbl.lambd == lam].sort_values("threshold")
        color = cmap(i / max(1, len(lams) - 1))
        axes[0].errorbar(sub.threshold, sub.desperate_mean,
                         yerr=[sub.desperate_mean - sub.desperate_lo,
                               sub.desperate_hi - sub.desperate_mean],
                         marker="o", capsize=3, color=color, label=f"λ = {lam}")
        axes[1].errorbar(sub.threshold, sub.nondesperate_mean,
                         yerr=[sub.nondesperate_mean - sub.nondesperate_lo,
                               sub.nondesperate_hi - sub.nondesperate_mean],
                         marker="o", capsize=3, color=color, label=f"λ = {lam}")
    axes[0].set_title("Desperate agents", loc="left")
    axes[1].set_title("Non-desperate agents", loc="left")
    for ax in axes:
        ax.set_xlabel("Relative desperation threshold (ν)")
        ax.set_ylabel("Proportion offending at step 25")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="white")
    fig.suptitle(f"Lambda sweep at β = 0, α = 1 ({tbl.n_reps.iloc[0]} reps per cell)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nWrote plot -> {out_png}")

    csv_path = out_png.replace(".png", "_summary.csv")
    tbl.to_csv(csv_path, index=False)
    print(f"Wrote table -> {csv_path}")


if __name__ == "__main__":
    main()
