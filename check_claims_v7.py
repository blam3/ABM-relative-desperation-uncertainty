"""Recompute the numeric and directional claims of the v7 Results and Supplement.

    python check_claims_v7.py --root ~/scratch/dmurd/rerun-2026-11-28 --out <dir>

Reads only outputs that record their own configuration (summary columns `Lambda`
and `Victim_Mode`, or `lambda` and `victim_mode` for the inequality sweep) and
refuses to run unless every row records lambda = 0.5 and victim mode "none".

Writes <out>/claims_values.csv (one row per computed quantity: id, value, MC 95%
interval) and replacement Tables S2-S6 as CSV and LaTeX tabular. Claim ids follow
reports/2026-11-28-decision-gate.md. Labels are assigned in the report, by the
rule stated there.

Monte Carlo 95% intervals: mean +/- 1.96 SD / sqrt(N) over replications. For
derived statistics (eta squared, coefficients, correlations, ratios) the interval
is the 2.5-97.5 percentile of a bootstrap that resamples replications within each
condition (B = 400, seed 0).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

B_BOOT = 400


def mc(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    m = v.mean()
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def require_config(df, lam_col, vm_col, label):
    if lam_col not in df or vm_col not in df:
        sys.exit(f"{label}: output does not record {lam_col}/{vm_col}; refusing to use it (UNVERIFIED)")
    if not np.allclose(df[lam_col].astype(float), 0.5) or set(df[vm_col]) != {"none"}:
        sys.exit(f"{label}: recorded lambda/victim mode not 0.5/none: "
                 f"{sorted(df[lam_col].unique())} {sorted(df[vm_col].unique())}")


def load_dir(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"{label}: no files match {pattern}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    require_config(df, "Lambda", "Victim_Mode", label)
    return df, len(files)


# --------------------------------------------------------------------- metamodel
def metamodel(df, alpha_coding="log2"):
    """Table 2 metamodel. alpha_coding='log2' enters log2(alpha) as the manuscript
    states; 'iota' enters -log2(alpha) as aggregate_results.py does."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    x = df[["Crime_Final", "Beta_Loc", "Alpha_Loc", "Threshold"]].copy()
    x["a"] = np.log2(x.Alpha_Loc) * (1 if alpha_coding == "log2" else -1)
    for src, dst in (("Beta_Loc", "zb"), ("a", "za"), ("Threshold", "zt")):
        x[dst] = (x[src] - x[src].mean()) / x[src].std()
    fit = smf.ols("Crime_Final ~ zb * za * zt", data=x).fit()
    a = anova_lm(fit, typ=3)
    ssr = a.loc["Residual", "sum_sq"]
    sst = ((x.Crime_Final - x.Crime_Final.mean()) ** 2).sum()
    tab = pd.DataFrame({"b": fit.params, "eta2": a["sum_sq"] / sst,
                        "partial_eta2": a["sum_sq"] / (a["sum_sq"] + ssr)})
    tab = tab.drop(["Intercept", "Residual"], errors="ignore")
    return tab, fit.rsquared


def boot_within(df, keys, stat, rng):
    """Resample replications within each condition; return list of stat values."""
    groups = [g.index.values for _, g in df.groupby(keys)]
    out = []
    for _ in range(B_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        out.append(stat(df.loc[idx].reset_index(drop=True)))
    return out


# --------------------------------------------------------------------- main sweep
def main_claims(D, rows, rng):
    K = ["Beta_Loc", "Alpha_Loc", "Threshold"]
    add = lambda cid, name, val, lo=np.nan, hi=np.nan: rows.append(
        dict(id=cid, quantity=name, value=val, lo=lo, hi=hi))

    tab, r2 = metamodel(D, "log2")
    tab_i, _ = metamodel(D, "iota")
    terms = {"zb": "beta", "za": "alpha(log2)", "zt": "nu", "zb:za": "beta x alpha",
             "zb:zt": "beta x nu", "za:zt": "alpha x nu", "zb:za:zt": "beta x alpha x nu"}

    def mm_stat(d):
        t, rr = metamodel(d, "log2")
        return np.concatenate([[rr], t.loc[list(terms), ["b", "eta2", "partial_eta2"]].values.ravel()])
    bs = np.array(boot_within(D, K, mm_stat, rng))
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    add("M1", "R2", r2, lo[0], hi[0])
    for i, (k, nm) in enumerate(terms.items()):
        for j, col in enumerate(["b", "eta2", "partial_eta2"]):
            p = 1 + 3 * i + j
            add("T2", f"{nm} {col}", tab.loc[k, col], lo[p], hi[p])
        add("T2", f"{nm} b (iota coding, aggregate_results.py)", tab_i.loc[k, "b"])

    for b in (-1, 0, 1):
        add("M2", f"median wealth step 25, beta={b}", *mc(D[D.Beta_Loc == b].Median_Wealth))
    for b in (-1, 0, 1):
        add("M3", f"Caught_Final (sticky) beta={b}", *mc(D[D.Beta_Loc == b].Caught_Final))
        add("M3", f"Crime_Final beta={b}", *mc(D[D.Beta_Loc == b].Crime_Final))
    add("M5", "frac in debt, all 45", *mc(D.Frac_Debt))
    u = D[(D.Beta_Loc == 0) & (D.Alpha_Loc == 1)]
    for t in sorted(D.Threshold.unique()):
        add("M6", f"frac in debt beta=0 alpha=1 nu={t}", *mc(u[u.Threshold == t].Frac_Debt))
    add("M7", "crime step 25, all 45, mean", *mc(D.Crime_Final))
    add("M7", "crime step 25, all 45, SD across runs", D.Crime_Final.std())
    s = u[u.Threshold == 0.2].Crime_Final
    add("M8", "crime beta=0 alpha=1 nu=.20", *mc(s)); add("M8", "SD", s.std())
    add("M9", "crime beta=0 alpha=1 nu=.05", *mc(u[u.Threshold == 0.05].Crime_Final))
    add("M10", "crime step 1, all 45", *mc(D.Crime_1))
    add("M11", "crime step 25, all 45", *mc(D.Crime_25))
    for b in (-1, 0, 1):
        add("M13", f"crime beta={b}", *mc(D[D.Beta_Loc == b].Crime_Final))
    m13 = D.groupby("Beta_Loc").Crime_Final.mean()
    add("M13", "fold reduction beta -1 / +1", m13[-1] / m13[1])
    for a in (0.5, 1.0, 2.0):
        add("M15", f"crime alpha={a}", *mc(D[D.Alpha_Loc == a].Crime_Final))
    add("M19", "desperate rate, all runs", *mc(D.Desperate_Rate_Final))
    add("M19", "desperate rate SD", D.Desperate_Rate_Final.std())
    add("M19", "non-desperate rate", *mc(D.NonDesperate_Rate_Final))
    add("M19", "non-desperate rate SD", D.NonDesperate_Rate_Final.std())
    D = D.assign(gap=D.Desperate_Rate_Final - D.NonDesperate_Rate_Final)
    for t in sorted(D.Threshold.unique()):
        g = D[D.Threshold == t]
        add("M20", f"gap nu={t}", *mc(g.gap))
        add("M21", f"desperate nu={t}", *mc(g.Desperate_Rate_Final))
        add("M21", f"non-desperate nu={t}", *mc(g.NonDesperate_Rate_Final))
    cond = D.groupby(K)[["LPM_b_Desperate", "LPM_b_Wealth"]].mean()
    add("M22", "LPM b desperate, mean over 45 conditions", cond.LPM_b_Desperate.mean())
    add("M22", "LPM b desperate, SD over conditions", cond.LPM_b_Desperate.std())
    add("M22", "LPM b wealth, mean over 45 conditions", cond.LPM_b_Wealth.mean())
    add("M22", "LPM b wealth, SD over conditions", cond.LPM_b_Wealth.std())
    add("M23", "conditions with LPM b desperate > 0", int((cond.LPM_b_Desperate > 0).sum()))
    add("M23", "conditions with LPM b wealth > 0", int((cond.LPM_b_Wealth > 0).sum()))
    for b in (-1, 0, 1):
        for a in (0.5, 1.0, 2.0):
            add("M24", f"gap cell beta={b} alpha={a}",
                *mc(D[(D.Beta_Loc == b) & (D.Alpha_Loc == a)].gap))
    for b in (-1, 0, 1):
        add("M26", f"gap beta={b} (over alpha, nu)", *mc(D[D.Beta_Loc == b].gap))
    for t in sorted(D.Threshold.unique()):
        add("M28", f"crime nu={t}", *mc(D[D.Threshold == t].Crime_Final))
    cm = D.groupby(K).agg(gap90=("Gap_Final", "mean"), crime=("Crime_Final", "mean"))
    cm_alt = D.assign(g=D.P90_Wealth - D.P10_Wealth).groupby(K).agg(g=("g", "mean"), c=("Crime_Final", "mean"))
    rr = boot_within(D, K, lambda d: d.groupby(K).agg(g=("Gap_Final", "mean"), c=("Crime_Final", "mean")).corr().iloc[0, 1], rng)
    add("M29", "r(wealth gap, crime) over 45 conditions", cm.gap90.corr(cm.crime), *np.percentile(rr, [2.5, 97.5]))
    add("M29", "r using P90-P10 columns (analyze_for_manuscript.py)", cm_alt.g.corr(cm_alt.c))
    for (b, a, t) in [(1, 0.5, 0.05), (-1, 2.0, 0.65)]:
        add("M31", f"crime beta={b} alpha={a} nu={t}",
            *mc(D[(D.Beta_Loc == b) & (D.Alpha_Loc == a) & (D.Threshold == t)].Crime_Final))
    cmm = D.groupby(K).Crime_Final.mean()
    add("M31", "fold range max/min condition", cmm.max() / cmm.min())
    add("M31", "argmin condition", str(cmm.idxmin())); add("M31", "argmax condition", str(cmm.idxmax()))

    def diff_mc(a_, b_):
        d = a_.mean() - b_.mean(); h = 1.96 * np.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
        return d, d - h, d + h
    for b in (-1, 1):
        sb = D[D.Beta_Loc == b]
        add("M34", f"crime alpha=2 minus alpha=0.5, beta={b}",
            *diff_mc(sb[sb.Alpha_Loc == 2].Crime_Final, sb[sb.Alpha_Loc == 0.5].Crime_Final))
    for b in (-1, 1):
        sb = D[D.Beta_Loc == b]
        add("M36", f"crime nu .65 minus .05, beta={b}",
            *diff_mc(sb[sb.Threshold == 0.65].Crime_Final, sb[sb.Threshold == 0.05].Crime_Final))
    for a in (2.0, 0.5):
        sa = D[D.Alpha_Loc == a]
        add("M36", f"crime nu .65 minus .05, alpha={a}",
            *diff_mc(sa[sa.Threshold == 0.65].Crime_Final, sa[sa.Threshold == 0.05].Crime_Final))
    for (b, a) in [(-1, 2.0), (1, 0.5)]:
        c = D[(D.Beta_Loc == b) & (D.Alpha_Loc == a)]
        add("M37", f"crime nu .65 minus .05, beta={b} alpha={a}",
            *diff_mc(c[c.Threshold == 0.65].Crime_Final, c[c.Threshold == 0.05].Crime_Final))
        means = c.groupby("Threshold").Crime_Final.mean()
        add("M37", f"argmax nu, beta={b} alpha={a}", float(means.idxmax()))
    mono = []
    for (b, a), c in D.groupby(["Beta_Loc", "Alpha_Loc"]):
        means = c.groupby("Threshold").Crime_Final.mean().values
        mono.append((b, a, bool(np.all(np.diff(means) > 0)), float(c.groupby("Threshold").Crime_Final.mean().idxmax())))
    for b, a, ok, am in mono:
        add("M38", f"crime rises monotonically in nu, beta={b} alpha={a} (argmax nu={am})", int(ok))
    return D


# --------------------------------------------------------------------- tables
def fmt(x, d=3):
    s = f"{x:.{d}f}"
    return "-0.000" if s in ("-0.000", "0.000") and x < 0 else s


def tex(s):
    return s.replace("-", "$-$")


def write_table(out, name, header_csv, rows_csv, header_tex, rows_tex, colspec):
    with open(os.path.join(out, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_csv)
        w.writerows(rows_csv)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\hline\n" + " & ".join(header_tex) + " \\\\\n\\hline\n")
        for r in rows_tex:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def sens_tables(SM, CV, INQ, out, rows):
    add = lambda cid, name, val, lo=np.nan, hi=np.nan: rows.append(
        dict(id=cid, quantity=name, value=val, lo=lo, hi=hi))
    # ---- S2
    SM = SM.assign(gap=SM.Desperate_Rate_Final - SM.NonDesperate_Rate_Final)
    rc, rt = [], []
    for (tau, th), g in sorted(SM.groupby(["Tau_Softmax", "Theta_Softmax"])):
        c, gg = mc(g.Crime_Final), mc(g.gap)
        add("S2", f"crime tau={tau} T={th}", *c); add("S2", f"gap tau={tau} T={th}", *gg)
        rc.append([tau, th] + [fmt(x) for x in c + gg])
        cells = [f"{tau:.1f}", f"{th:.1f}", tex(f"{fmt(c[0])} [{fmt(c[1])}, {fmt(c[2])}]"),
                 tex(f"{fmt(gg[0])} [{fmt(gg[1])}, {fmt(gg[2])}]")]
        if tau == 3.0 and th == 1.5:
            cells = ["\\textbf{" + x + "}" for x in cells]
        rt.append(cells)
    write_table(out, "table_S2", "tau,theta_noise,mean_crime_rate,crime_ci_lo,crime_ci_hi,mean_desperation_gap,gap_ci_lo,gap_ci_hi".split(","),
                rc, ["$\\tau$", "$\\theta_{noise}$", "Mean crime rate (95\\% CI)", "Mean desperation gap (95\\% CI)"], rt, "rrcc")
    for tau in (1.0, 3.0, 5.0):
        add("S50", f"marginal crime tau={tau}", *mc(SM[SM.Tau_Softmax == tau].Crime_Final))
    for th in (0.5, 1.5, 3.0):
        add("S51", f"marginal crime T={th}", *mc(SM[SM.Theta_Softmax == th].Crime_Final))
    # ---- S3, S4, S5
    steps = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    r3c, r3t, r4c, r4t, r5c, r5t = [], [], [], [], [], []
    for (b, t), g in sorted(CV.groupby(["Beta_Loc", "Threshold"])):
        means = [g[f"Crime_{s}"].mean() for s in steps]
        r3c.append([b, t] + [f"{m:.3f}" for m in means])
        r3t.append([tex(f"{b:+d}" if b else "0"), f"{t:.2f}"] + [f"{m:.3f}" for m in means])
        s25, s50 = mc(g.Crime_25), mc(g.Crime_50)
        dlt = s50[0] - s25[0]; pct = 100 * dlt / s25[0]
        add("S4", f"step25 beta={b} nu={t}", *s25); add("S4", f"step50 beta={b} nu={t}", *s50)
        add("S4", f"pct change beta={b} nu={t}", pct)
        r4c.append([b, t] + [fmt(x) for x in s25 + s50] + [f"{dlt:+.3f}", f"{pct:+.1f}"])
        r4t.append([tex(f"{b:+d}" if b else "0"), f"{t:.2f}", f"{fmt(s25[0])} [{fmt(s25[1])}, {fmt(s25[2])}]",
                    f"{fmt(s50[0])} [{fmt(s50[1])}, {fmt(s50[2])}]", tex(f"{dlt:+.3f}"), tex(f"{pct:+.1f}") + "\\%"])
        pre = np.abs(np.diff(g[[f"Crime_{s}" for s in range(20, 26)]].values, axis=1)).mean()
        post = np.abs(np.diff(g[[f"Crime_{s}" for s in range(25, 51)]].values, axis=1)).mean()
        add("S5", f"pre beta={b} nu={t}", pre); add("S5", f"post beta={b} nu={t}", post); add("S5", f"ratio beta={b} nu={t}", post / pre)
        r5c.append([b, t, f"{pre:.5f}", f"{post:.5f}", f"{post / pre:.2f}"])
        r5t.append([tex(f"{b:+d}" if b else "0"), f"{t:.2f}", f"{pre:.5f}", f"{post:.5f}", f"{post / pre:.2f}$\\times$"])
    write_table(out, "table_S3", ["beta", "nu"] + [f"s{s:02d}" for s in steps], r3c,
                ["$\\beta$", "$\\nu$"] + [f"s{s:02d}" for s in steps], r3t, "rr" + "r" * len(steps))
    write_table(out, "table_S4", "beta,nu,step25_mean,step25_lo,step25_hi,step50_mean,step50_lo,step50_hi,change,pct_change".split(","), r4c,
                ["$\\beta$", "$\\nu$", "Step 25 (95\\% CI)", "Step 50 (95\\% CI)", "Change", "\\% Change"], r4t, "rrccrr")
    write_table(out, "table_S5", "beta,nu,mean_abs_change_20_25,mean_abs_change_25_50,ratio_post_pre".split(","), r5c,
                ["$\\beta$", "$\\nu$", "Mean $|\\Delta|$ steps 20--25", "Mean $|\\Delta|$ steps 25--50", "Ratio (post/pre)"], r5t, "rrrrr")
    # orderings at every step
    for s in range(1, 51):
        for t in (0.05, 0.35, 0.65):
            m = [CV[(CV.Beta_Loc == b) & (CV.Threshold == t)][f"Crime_{s}"].mean() for b in (-1, 0, 1)]
            if not (m[0] > m[1] > m[2]):
                add("S59", f"beta ordering broken at step {s}, nu={t}", 1)
        for b in (-1, 0, 1):
            m = [CV[(CV.Beta_Loc == b) & (CV.Threshold == t)][f"Crime_{s}"].mean() for t in (0.05, 0.35, 0.65)]
            if not (m[0] < m[1] < m[2]):
                add("S59", f"nu ordering broken at step {s}, beta={b}", 1)
    CV = CV.assign(gap=CV.Desperate_Rate_Final - CV.NonDesperate_Rate_Final)
    for (b, t), g in sorted(CV.groupby(["Beta_Loc", "Threshold"])):
        add("C", f"convergence gap at step 50, beta={b} nu={t}", *mc(g.gap))
    # ---- S6 (inequality)
    lab = {("rank", 0.2): "Rank, nu = 0.20", ("rank", 0.5): "Rank, nu = 0.50", ("median_ratio", None): "Median ratio"}
    r6c, r6t = [], []
    INQ = INQ.copy(); INQ["thresh"] = INQ.thresh.where(INQ.thresh.notna(), None)
    for rule_key in [("rank", 0.2), ("rank", 0.5), ("median_ratio", None)]:
        for sp in ("low", "mid", "high"):
            g = INQ[(INQ.spread == sp) & (INQ["rule"] == rule_key[0])]
            if rule_key[1] is not None:
                g = g[np.isclose(g.thresh.astype(float), rule_key[1])]
            n = len(g)
            traj_c = g[[f"crime_{s}" for s in range(1, 26)]].mean(axis=1)
            traj_d = g[[f"desperate_{s}" for s in range(1, 26)]].mean(axis=1)
            vals = [traj_c.mean(), g.crime_25.mean(), traj_d.mean(), g.desperate_25.mean(), g.gap_25.mean()]
            for nm, col in zip(["offending 1-25", "offending 25", "desperate 1-25", "desperate 25", "gap 25"],
                               [traj_c, g.crime_25, traj_d, g.desperate_25, g.gap_25]):
                add("S6", f"{nm} {sp} {lab[rule_key]} (n={n})", *mc(col))
            add("S6", f"crime_1 {sp} {lab[rule_key]}", *mc(g.crime_1))
            add("S6", f"init_gap {sp} {lab[rule_key]}", *mc(g.init_gap))
            name = {"low": "Low", "mid": "Medium", "high": "High"}[sp]
            r6c.append([name, lab[rule_key]] + [f"{v:.3f}" for v in vals[:4]] + [f"{vals[4]:.0f}"])
            r6t.append([name, lab[rule_key].replace("nu", "$\\nu$")] + [f"{v:.3f}" for v in vals[:4]] + [f"{vals[4]:.0f}"])
    write_table(out, "table_S6", "dispersion,rule,offending_steps_1_25,offending_step_25,desperate_share_steps_1_25,desperate_share_step_25,wealth_gap_p90_p10_step_25".split(","),
                r6c, ["Dispersion", "Rule", "Offending (steps 1--25)", "Offending (step 25)", "Desperate share (steps 1--25)",
                      "Desperate share (step 25)", "Wealth gap (90th $-$ 10th percentile)"], r6t, "llrrrrr")
    # crime ordering crossover step under each rule (first step low > high)
    for rule_key in [("rank", 0.2), ("rank", 0.5), ("median_ratio", None)]:
        g = INQ[INQ["rule"] == rule_key[0]]
        if rule_key[1] is not None:
            g = g[np.isclose(g.thresh.astype(float), rule_key[1])]
        lo_ = g[g.spread == "low"]; hi_ = g[g.spread == "high"]
        cross = [s for s in range(1, 26) if lo_[f"crime_{s}"].mean() > hi_[f"crime_{s}"].mean()]
        add("S64", f"first step low > high crime, {lab[rule_key]}", cross[0] if cross else -1)
        add("S64", f"high/low crime ratio at step 1, {lab[rule_key]}", hi_.crime_1.mean() / lo_.crime_1.mean())
        add("S64", f"high > low at step 25, {lab[rule_key]}", int(hi_.crime_25.mean() > lo_.crime_25.mean()))
        for sp in ("low", "mid", "high"):
            gg = g[g.spread == sp]
            add("S64", f"gap25/init_gap {sp} {lab[rule_key]}", gg.gap_25.mean() / gg.init_gap.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    D, nf = load_dir(os.path.join(args.root, "final_batch_run_lambda0.5", "summary_*.csv"), "main sweep")
    print(f"main sweep: {nf} files, {len(D)} runs")
    SM, nf = load_dir(os.path.join(args.root, "sensitivity_softmax_lambda0.5", "summary_*.csv"), "softmax")
    print(f"softmax: {nf} files, {len(SM)} runs")
    CV, nf = load_dir(os.path.join(args.root, "sensitivity_convergence_lambda0.5", "summary_*.csv"), "convergence")
    print(f"convergence: {nf} files, {len(CV)} runs")
    INQ = pd.read_csv(os.path.join(args.root, "inequality_sweep", "summary.csv"))
    require_config(INQ, "lambda", "victim_mode", "inequality")
    print(f"inequality: {len(INQ)} runs")
    main_claims(D, rows, rng)
    sens_tables(SM, CV, INQ, args.out, rows)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "claims_values.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_rows", 1000, "display.max_colwidth", 80)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
