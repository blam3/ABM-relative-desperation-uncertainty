"""Compute every summary statistic needed to rewrite the Results section.

Usage: python analyze_for_manuscript.py [--dir <data dir>]
Prints statistics; does not write files.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd


def ci95(vals):
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = rng.choice(np.asarray(vals), size=(2000, n), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",
                    default=os.path.expanduser("~/scratch/dmurd/final_batch_run_lambda0.5"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "summary_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(files)} files, {len(df)} runs\n")

    print("== Overall crime ==")
    m = df.Crime_Final.mean(); s = df.Crime_Final.std()
    print(f"  crime overall M={m:.3f}, SD={s:.3f}")

    print("\n== By beta (mean, 95% CI) ==")
    for b in sorted(df.Beta_Loc.unique()):
        sub = df[df.Beta_Loc == b].Crime_Final
        lo, hi = ci95(sub.values)
        print(f"  beta={b}: M={sub.mean():.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n== By alpha (mean, 95% CI) ==")
    for a in sorted(df.Alpha_Loc.unique()):
        sub = df[df.Alpha_Loc == a].Crime_Final
        lo, hi = ci95(sub.values)
        print(f"  alpha={a}: M={sub.mean():.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n== By threshold nu (mean, 95% CI) ==")
    for t in sorted(df.Threshold.unique()):
        sub = df[df.Threshold == t].Crime_Final
        lo, hi = ci95(sub.values)
        print(f"  nu={t}: M={sub.mean():.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n== Desperate vs non-desperate (pooled) ==")
    d = df.Desperate_Rate_Final
    nd = df.NonDesperate_Rate_Final
    diff = d - nd
    pooled_sd = np.sqrt((d.var(ddof=1) + nd.var(ddof=1)) / 2)
    print(f"  desperate       M={d.mean():.3f}, SD={d.std():.3f}")
    print(f"  non-desperate   M={nd.mean():.3f}, SD={nd.std():.3f}")
    print(f"  paired gap      M={diff.mean():.3f}, SD={diff.std():.3f}")
    print(f"  Cohen d (pooled)= {(d.mean()-nd.mean())/pooled_sd:.3f}")

    print("\n== Desperate/non-desperate by nu ==")
    for t in sorted(df.Threshold.unique()):
        sub = df[df.Threshold == t]
        print(f"  nu={t}: desp={sub.Desperate_Rate_Final.mean():.3f}, "
              f"nondesp={sub.NonDesperate_Rate_Final.mean():.3f}, "
              f"gap={(sub.Desperate_Rate_Final.mean()-sub.NonDesperate_Rate_Final.mean()):.3f}")

    print("\n== Desperation gap by beta x alpha ==")
    gap_cells = (df.groupby(["Beta_Loc", "Alpha_Loc"])
                   .apply(lambda g: g.Desperate_Rate_Final.mean() - g.NonDesperate_Rate_Final.mean(),
                          include_groups=False))
    print(gap_cells.round(3).to_string())
    print(f"\n  Min gap: {gap_cells.min():.3f} at {gap_cells.idxmin()}")
    print(f"  Max gap: {gap_cells.max():.3f} at {gap_cells.idxmax()}")

    print("\n== Marginal gap by beta (averaged over alpha, nu) ==")
    for b in sorted(df.Beta_Loc.unique()):
        sub = df[df.Beta_Loc == b]
        g = sub.Desperate_Rate_Final - sub.NonDesperate_Rate_Final
        print(f"  beta={b}: gap M={g.mean():.3f}")

    print("\n== LPM: desperate coef vs wealth coef ==")
    if "LPM_b_Desperate" in df.columns:
        d_cond = df.groupby(["Beta_Loc", "Alpha_Loc", "Threshold"]).LPM_b_Desperate.mean()
        w_cond = df.groupby(["Beta_Loc", "Alpha_Loc", "Threshold"]).LPM_b_Wealth.mean()
        print(f"  LPM_b_Desperate: M={d_cond.mean():.3f}, SD={d_cond.std():.3f}, sign>0 in {(d_cond>0).sum()}/45")
        print(f"  LPM_b_Wealth   : M={w_cond.mean():.3f}, SD={w_cond.std():.3f}")
        print(f"  ratio |D|/|W|  : {abs(d_cond.mean())/max(1e-9, abs(w_cond.mean())):.2f}x")

    print("\n== Wealth summary ==")
    if "Median_Wealth" in df.columns:
        for b in sorted(df.Beta_Loc.unique()):
            sub = df[df.Beta_Loc == b].Median_Wealth
            print(f"  beta={b}: median wealth M={sub.mean():.1f}")
        print(f"  Frac in debt overall: M={df.Frac_Debt.mean():.3f}")

    print("\n== Response surface extremes ==")
    cm = (df.groupby(["Beta_Loc", "Alpha_Loc", "Threshold"])
            .Crime_Final.agg(["mean", "std", "count"])
            .reset_index())
    boot = []
    for _, row in cm.iterrows():
        sub = df[(df.Beta_Loc == row.Beta_Loc) & (df.Alpha_Loc == row.Alpha_Loc) &
                 (df.Threshold == row.Threshold)].Crime_Final.values
        lo, hi = ci95(sub)
        boot.append((row.Beta_Loc, row.Alpha_Loc, row.Threshold, row["mean"], lo, hi))
    boot_df = pd.DataFrame(boot, columns=["Beta", "Alpha", "Threshold", "M", "lo", "hi"])
    argmin = boot_df.iloc[boot_df.M.argmin()]
    argmax = boot_df.iloc[boot_df.M.argmax()]
    print(f"  MIN: beta={argmin.Beta}, alpha={argmin.Alpha}, nu={argmin.Threshold} -> "
          f"M={argmin.M:.3f} 95%CI[{argmin.lo:.3f},{argmin.hi:.3f}]")
    print(f"  MAX: beta={argmax.Beta}, alpha={argmax.Alpha}, nu={argmax.Threshold} -> "
          f"M={argmax.M:.3f} 95%CI[{argmax.lo:.3f},{argmax.hi:.3f}]")
    print(f"  ratio MAX/MIN = {argmax.M / argmin.M:.2f}x")

    print("\n== Beta x alpha two-way slopes (alpha effect at each beta) ==")
    for b in sorted(df.Beta_Loc.unique()):
        s05 = df[(df.Beta_Loc == b) & (df.Alpha_Loc == 0.5)].Crime_Final.mean()
        s10 = df[(df.Beta_Loc == b) & (df.Alpha_Loc == 1.0)].Crime_Final.mean()
        s20 = df[(df.Beta_Loc == b) & (df.Alpha_Loc == 2.0)].Crime_Final.mean()
        print(f"  beta={b}: alpha=0.5 -> {s05:.3f}, alpha=1 -> {s10:.3f}, alpha=2 -> {s20:.3f}, delta={s20-s05:+.3f}")

    print("\n== Beta x nu slopes ==")
    for b in sorted(df.Beta_Loc.unique()):
        low = df[(df.Beta_Loc == b) & (df.Threshold == 0.05)].Crime_Final.mean()
        high = df[(df.Beta_Loc == b) & (df.Threshold == 0.65)].Crime_Final.mean()
        print(f"  beta={b}: nu=0.05 -> {low:.3f}, nu=0.65 -> {high:.3f}, delta={high-low:+.3f}")

    print("\n== Alpha x nu slopes ==")
    for a in sorted(df.Alpha_Loc.unique()):
        low = df[(df.Alpha_Loc == a) & (df.Threshold == 0.05)].Crime_Final.mean()
        high = df[(df.Alpha_Loc == a) & (df.Threshold == 0.65)].Crime_Final.mean()
        print(f"  alpha={a}: nu=0.05 -> {low:.3f}, nu=0.65 -> {high:.3f}, delta={high-low:+.3f}")

    print("\n== P90-P10 correlation with crime ==")
    df["P90_P10_Gap"] = df.P90_Wealth - df.P10_Wealth
    cm2 = df.groupby(["Beta_Loc", "Alpha_Loc", "Threshold"]).agg(
        p90_p10=("P90_P10_Gap", "mean"),
        crime=("Crime_Final", "mean")).reset_index()
    r = cm2.p90_p10.corr(cm2.crime)
    print(f"  Pearson r (45 conditions) = {r:.3f}")

    print("\n== Trajectory endpoints step 1 vs step 25 (overall) ==")
    if "Crime_1" in df.columns and "Crime_25" in df.columns:
        print(f"  step 1  M={df.Crime_1.mean():.3f}")
        print(f"  step 25 M={df.Crime_25.mean():.3f}")


if __name__ == "__main__":
    main()
