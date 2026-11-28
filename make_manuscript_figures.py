"""
Reproduce all manuscript figures using the updated full-sweep data.

Data source: $DMURD_FIGURE_DATA (default: $HOME/scratch/dmurd/final_batch_run_v2/)
Output:      analysis/figures/

Usage:
    conda run -n dmurd python make_manuscript_figures.py

Terminology note: the parameter formerly described as "optimism" is labelled
"uncertainty-seeking" throughout (beta = -1 condition).
"""

import glob, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH = os.environ.get(
    "DMURD_FIGURE_DATA",
    os.path.expanduser("~/scratch/dmurd/final_batch_run_lambda0.5"))
OUT = os.environ.get(
    "DMURD_FIGURE_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "figures"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, green
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"

BETA_C  = {-1: S2, 0: S1, 1: S3}
BETA_L  = {
    -1: "β = −1 (Uncertainty-seeking)",
    0:  "β = 0 (Uncertainty neutral)",
    1:  "β = 1 (Uncertainty-averse)",
}
ALPHA_C = {0.5: S3, 1.0: S1, 2.0: S2}
ALPHA_L = {0.5: "α = 0.5 (Insensitive)",
           1.0: "α = 1.0 (No insensitivity)",
           2.0: "α = 2.0 (Oversensitive)"}
THRESH_COLORS = {0.05: "#9b59b6", 0.20: "#3498db", 0.35: "#2ecc71",
                 0.50: "#e67e22", 0.65: "#e74c3c"}
THRESH_LABELS = {v: f"ν = {v:.2f}" for v in [0.05, 0.20, 0.35, 0.50, 0.65]}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "figure.dpi": 200,
})

def tidy(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:  ax.set_title(title, loc="left", pad=8)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

def ci95(series):
    """Bootstrap 95% CI (percentile) for a mean."""
    n = len(series)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = rng.choice(series.values, size=(2000, n), replace=True).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
files = sorted(glob.glob(os.path.join(SCRATCH, "summary_*.csv")))
assert files, f"No summary_*.csv in {SCRATCH}"
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  {len(files)} condition files, {len(df)} replications")

# Convenience derived columns
df["iota"] = -np.log2(df["Alpha_Loc"])   # likelihood insensitivity index
df["Desperation_Gap"] = df["Desperate_Rate_Final"] - df["NonDesperate_Rate_Final"]
df["P90_P10_Gap"] = df["P90_Wealth"] - df["P10_Wealth"]

# Condition means (one row per condition)
COND_KEYS = ["Beta_Loc", "Alpha_Loc", "Threshold"]
cm = (df.groupby(COND_KEYS)
        .agg(
            Crime_Final_mean=("Crime_Final", "mean"),
            Crime_Final_std=("Crime_Final", "std"),
            P90_P10_mean=("P90_P10_Gap", "mean"),
            Desperate_Rate_mean=("Desperate_Rate_Final", "mean"),
            NonDesperate_Rate_mean=("NonDesperate_Rate_Final", "mean"),
            Desperation_Gap_mean=("Desperation_Gap", "mean"),
            Desperation_Gap_std=("Desperation_Gap", "std"),
        )
        .reset_index())

betas   = sorted(df["Beta_Loc"].unique())
alphas  = sorted(df["Alpha_Loc"].unique())
threshs = sorted(df["Threshold"].unique())

print("  Conditions:")
print(f"    β levels: {betas}")
print(f"    α levels: {alphas}")
print(f"    ν levels: {threshs}")

# ---------------------------------------------------------------------------
# Shared y-axis label / range for "proportion offending"
# ---------------------------------------------------------------------------
YLAB = "Proportion offending at step 25"


# ============================================================
# Figure 1 – Main effects: raincloud plots
# Hypothesis: (H1a) Greater uncertainty aversion reduces offending;
#             (H1b) Greater likelihood insensitivity amplifies offending;
#             (H2)  A higher relative desperation threshold raises aggregate offending.
# ============================================================
print("Figure 1 …")

def raincloud(ax, groups, data_col, color_map, group_labels, jitter_seed=1):
    """Violin + strip jitter side by side (right-half violin only)."""
    rng = np.random.default_rng(jitter_seed)
    positions = range(len(groups))
    for xi, g in enumerate(groups):
        vals = data_col[g].dropna().values
        color = color_map[g]
        # violin (full)
        vp = ax.violinplot([vals], positions=[xi], widths=0.55,
                           showmeans=False, showextrema=False, showmedians=False)
        for body in vp["bodies"]:
            # keep only right half
            m = np.mean(body.get_paths()[0].vertices[:, 0])
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], m, np.inf)
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
            body.set_linewidth(0.5)
        # jitter strip (left of violin)
        jitter = rng.uniform(-0.18, -0.03, size=len(vals))
        ax.scatter(xi + jitter, vals, s=4, color=color, alpha=0.4, linewidths=0)
        # mean + 95% CI
        lo, hi = ci95(pd.Series(vals))
        mn = vals.mean()
        ax.plot([xi - 0.22, xi - 0.22], [lo, hi], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter([xi - 0.22], [mn], s=22, color=color, zorder=5,
                   edgecolor=SURF, linewidth=0.8)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

# Left: β
groups_b = betas
data_b = {b: df.loc[df.Beta_Loc == b, "Crime_Final"] for b in groups_b}
raincloud(axes[0], groups_b, data_b, BETA_C, BETA_L)
axes[0].set_xticks(range(len(groups_b)))
axes[0].set_xticklabels(["−1\n(Uncertainty-\nseeking)", "0\n(Uncertainty\nneutral)", "1\n(Uncertainty-\naverse)"])
tidy(axes[0], "Uncertainty aversion (β)",
     "Uncertainty aversion (β)", YLAB)

# Center: α
groups_a = alphas
data_a = {a: df.loc[df.Alpha_Loc == a, "Crime_Final"] for a in groups_a}
raincloud(axes[1], groups_a, data_a, ALPHA_C, ALPHA_L)
axes[1].set_xticks(range(len(groups_a)))
axes[1].set_xticklabels(["0.5\n(Insensitive)", "1\n(No insensitivity)", "2\n(Oversensitive)"])
tidy(axes[1], "Likelihood insensitivity (α)",
     "Likelihood insensitivity (α)")

# Right: ν
groups_t = threshs
data_t = {t: df.loc[df.Threshold == t, "Crime_Final"] for t in groups_t}
t_colors = {t: plt.cm.plasma(i / (len(threshs) - 1)) for i, t in enumerate(threshs)}
raincloud(axes[2], groups_t, data_t, t_colors, THRESH_LABELS)
axes[2].set_xticks(range(len(groups_t)))
axes[2].set_xticklabels([f"{t:.2f}" for t in threshs])
tidy(axes[2], "Desperation threshold (ν)",
     "Relative desperation threshold (ν)")

for ax in axes:
    ax.set_ylim(0, 0.4)

save(fig, "fig1_main_effects.png")


# ============================================================
# Figure 2 – Desperate vs. non-desperate offending by threshold
# Hypothesis: Agents below the relative desperation threshold offend at
#             substantially higher rates than non-desperate agents of comparable wealth.
# ============================================================
print("Figure 2 …")

fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), sharey=True)
fig2_upper_ci = []
for ax, th in zip(axes, threshs):
    sub = df[df.Threshold == th]
    # per-replication desperate and non-desperate rates → distribution
    desp = sub["Desperate_Rate_Final"].values
    nond = sub["NonDesperate_Rate_Final"].values
    # box-style summary: mean ± 95% CI
    for xi, (vals, label, color) in enumerate([
            (desp, "Desperate", S2),
            (nond, "Non-\ndesperate", S1)]):
        mn = vals.mean()
        lo, hi = ci95(pd.Series(vals))
        fig2_upper_ci.append(hi)
        ax.bar(xi, mn, color=color, alpha=0.75, width=0.55, zorder=2)
        ax.errorbar(xi, mn, yerr=[[mn - lo], [hi - mn]],
                    fmt="none", color=INK, capsize=4, lw=1.2, zorder=3)
        ax.text(xi, hi + 0.006, f"{mn:.3f}", ha="center", fontsize=7, color=INK2)
    tidy(ax, f"ν = {th:.2f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Desperate", "Non-\ndesperate"], fontsize=7.5)

axes[0].set_ylabel(YLAB)
fig2_ymax = math.ceil(max(fig2_upper_ci) / 0.05) * 0.05
axes[0].set_ylim(0, fig2_ymax)
print(f"  Figure 2 y-limit: 0 to {fig2_ymax:.2f} "
      f"(max upper CI = {max(fig2_upper_ci):.4f})")
save(fig, "fig2_desperate_vs_nondesperate.png")


# ============================================================
# Figure 3 – Desperation gap modulated by β and α
# Hypothesis: The desperation effect (desperate − non-desperate offending gap)
#             is largest under uncertainty-seeking + high insensitivity and
#             smallest under uncertainty aversion + high insensitivity.
# ============================================================
print("Figure 3 …")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))

# Left panel: heatmap of mean desperation gap by β × α
gap_mat = (cm.groupby(["Beta_Loc", "Alpha_Loc"])["Desperation_Gap_mean"]
             .mean()
             .unstack("Alpha_Loc")
             .loc[betas, alphas])

import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=gap_mat.values.min(), vmax=gap_mat.values.max())
im = axes[0].imshow(gap_mat.values, aspect="auto", cmap="YlOrRd", norm=norm,
                    origin="upper")
axes[0].set_xticks(range(len(alphas)))
axes[0].set_xticklabels(["0.5\n(Insensitive)", "1.0\n(No insensitivity)", "2.0\n(Oversensitive)"])
axes[0].set_yticks(range(len(betas)))
axes[0].set_yticklabels(["−1 (Uncertainty-\n     seeking)", "0 (Uncertainty\n     neutral)", "+1 (Uncertainty-\n     averse)"])
axes[0].set_xlabel("Likelihood insensitivity (α)")
axes[0].set_ylabel("Uncertainty aversion (β)")
for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        val = gap_mat.loc[b, a]
        axes[0].text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=8.5, color="white" if val > 0.12 else INK)
fig.colorbar(im, ax=axes[0], label="Desperation gap\n(Proportion offending)", fraction=0.046)
tidy(axes[0], "Desperation gap by β × α")
axes[0].grid(False)

# Right panel: marginal β effect on desperation gap (averaged over α and ν)
beta_gap = (df.groupby("Beta_Loc")["Desperation_Gap"]
              .apply(lambda x: pd.Series({
                  "mean": x.mean(),
                  "lo": ci95(x)[0],
                  "hi": ci95(x)[1],
              }))
              .unstack())

for xi, b in enumerate(betas):
    mn, lo, hi = beta_gap.loc[b, "mean"], beta_gap.loc[b, "lo"], beta_gap.loc[b, "hi"]
    axes[1].errorbar(xi, mn, yerr=[[mn - lo], [hi - mn]],
                     fmt="o", color=BETA_C[b], ms=7, capsize=5, lw=1.5,
                     markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
    axes[1].text(xi, hi + 0.003, f"{mn:.3f}", ha="center", fontsize=7.5, color=INK2)

axes[1].set_xticks(range(len(betas)))
axes[1].set_xticklabels(["−1\n(Uncertainty-\nseeking)", "0\n(Uncertainty\nneutral)", "+1\n(Uncertainty-\naverse)"])
axes[1].set_xlim(-0.5, 2.5)
fig3_upper = float(beta_gap["hi"].max())
# +.003 headroom to keep numeric labels above points from clipping
fig3_ymax = math.ceil((fig3_upper + 0.003) / 0.005) * 0.005
axes[1].set_ylim(0, fig3_ymax)
print(f"  Figure 3 right-panel y-limit: 0 to {fig3_ymax:.3f} "
      f"(max upper CI = {fig3_upper:.4f})")
tidy(axes[1], "Marginal effect of β on the desperation gap",
     "Uncertainty aversion (β)",
     "Desperation gap\n(Difference in offending rates between\ndesperate and non-desperate agents)")
axes[1].grid(False)

save(fig, "fig3_desperation_gap_modulation.png")


# ============================================================
# Figure 4 – Wealth inequality and aggregate offending
# Hypothesis: Conditions with greater wealth inequality (P90–P10 gap) show
#             higher aggregate offending, with uncertainty-seeking amplifying
#             the relationship.
# ============================================================
print("Figure 4 …")

fig, ax = plt.subplots(figsize=(5.8, 4.2))

for b in betas:
    sub = cm[cm.Beta_Loc == b]
    ax.scatter(sub["P90_P10_mean"], sub["Crime_Final_mean"],
               color=BETA_C[b], label=BETA_L[b], s=40,
               edgecolor=SURF, linewidth=0.7, zorder=3, alpha=0.9)

# Pearson's r across all 45 conditions
r = np.corrcoef(cm["P90_P10_mean"], cm["Crime_Final_mean"])[0, 1]
ax.text(0.04, 0.93, f"Pearson's r = {r:.3f} (all 45 conditions)",
        transform=ax.transAxes, fontsize=8, color=INK2,
        bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"))

tidy(ax, None,
     "P90 − P10 Wealth gap",
     YLAB)
ax.legend(loc="lower right", fontsize=8, handlelength=1.2,
          frameon=True, facecolor="white", edgecolor="white")

save(fig, "fig4_inequality_offending.png")


# ============================================================
# Figure 5 – Offending trajectories over simulation time
# Hypothesis: Offending stabilises within the 25-step window; the desperation
#             threshold determines the equilibrium level, while uncertainty
#             aversion shifts the entire trajectory down.
# ============================================================
print("Figure 5 …")

step_cols  = [f"Crime_{s}" for s in range(1, 26)]
steps      = list(range(1, 26))

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)

fig5_upper_bands = []
for ax, b in zip(axes, betas):
    sub = df[df.Beta_Loc == b]
    for th in threshs:
        g = sub[sub.Threshold == th]
        traj = g[step_cols].values        # shape (n_reps, 25)
        mn   = traj.mean(axis=0)
        lo   = np.percentile(traj, 2.5, axis=0)
        hi   = np.percentile(traj, 97.5, axis=0)
        fig5_upper_bands.append(hi.max())
        color = THRESH_COLORS[th]
        ax.plot(steps, mn, color=color, label=f"ν = {th:.2f}", lw=1.6)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.axvline(25, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(1, 27)
    tidy(ax, BETA_L[b], "Simulation step",
         "Proportion offending" if b == betas[0] else None)

fig5_ymax = math.ceil(max(fig5_upper_bands) / 0.05) * 0.05
axes[0].set_ylim(0, fig5_ymax)
print(f"  Figure 5 y-limit: 0 to {fig5_ymax:.2f} "
      f"(max upper band = {max(fig5_upper_bands):.4f})")
handles = [Line2D([0], [0], color=THRESH_COLORS[t], lw=1.8, label=f"ν = {t:.2f}")
           for t in threshs]
axes[-1].legend(handles=handles, frameon=False, loc="upper right", fontsize=7.5)

save(fig, "fig5_trajectories.png")


# ============================================================
# Figure 5 (alternative) – same content, β stacked vertically
# so that β appears on the y-axis (rows) rather than as columns.
# ============================================================
print("Figure 5 (alternative, β on y-axis) …")

fig, axes = plt.subplots(3, 1, figsize=(6.0, 8.4), sharex=True)

fig5alt_upper_bands = []
for ax, b in zip(axes, betas):
    sub = df[df.Beta_Loc == b]
    for th in threshs:
        g = sub[sub.Threshold == th]
        traj = g[step_cols].values
        mn = traj.mean(axis=0)
        lo = np.percentile(traj, 2.5, axis=0)
        hi = np.percentile(traj, 97.5, axis=0)
        fig5alt_upper_bands.append(hi.max())
        color = THRESH_COLORS[th]
        ax.plot(steps, mn, color=color, label=f"ν = {th:.2f}", lw=1.6)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.axvline(25, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(1, 27)
    tidy(ax, BETA_L[b], None, "Proportion offending")

axes[-1].set_xlabel("Simulation step")
fig5alt_ymax = math.ceil(max(fig5alt_upper_bands) / 0.05) * 0.05
for ax in axes:
    ax.set_ylim(0, fig5alt_ymax)
print(f"  Figure 5 (alt) y-limit: 0 to {fig5alt_ymax:.2f} "
      f"(max upper band = {max(fig5alt_upper_bands):.4f})")

handles = [Line2D([0], [0], color=THRESH_COLORS[t], lw=1.8, label=f"ν = {t:.2f}")
           for t in threshs]
axes[0].legend(handles=handles, frameon=False, loc="upper right", fontsize=7.5)

save(fig, "fig5alt_trajectories_beta_rows.png")


# ============================================================
# Figure 5 (heatmap) – step × β heatmap of mean offending, one panel per ν
# so β appears as a numeric y-axis rather than a panel label.
# ============================================================
print("Figure 5 (heatmap, β on y-axis) …")

fig, axes = plt.subplots(1, len(threshs), figsize=(13.5, 3.2),
                         sharey=True, sharex=True)

heat_mats = []
for th in threshs:
    rows = []
    for b in betas:
        g = df[(df.Beta_Loc == b) & (df.Threshold == th)]
        rows.append(g[step_cols].values.mean(axis=0))
    heat_mats.append(np.vstack(rows))        # shape (len(betas), 25)

vmin = min(m.min() for m in heat_mats)
vmax = max(m.max() for m in heat_mats)

im = None
for ax, th, mat in zip(axes, threshs, heat_mats):
    im = ax.imshow(mat, aspect="auto", cmap="magma",
                   vmin=vmin, vmax=vmax, origin="lower",
                   extent=[0.5, 25.5, -0.5, len(betas) - 0.5])
    tidy(ax, f"ν = {th:.2f}", "Simulation step",
         "Uncertainty aversion (β)" if th == threshs[0] else None)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(["−1" if b == -1 else ("0" if b == 0 else "+1") for b in betas])
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.grid(False)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("Proportion offending")
print(f"  Figure 5 (heatmap) colour range: {vmin:.4f} to {vmax:.4f}")

path = os.path.join(OUT, "fig5heatmap_trajectories_beta_y.png")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {path}")


# ============================================================
# Figure 6 – Three-way interaction: β × α × ν
# Hypothesis: The effect of the desperation threshold on offending is steeper
#             under uncertainty-seeking than under uncertainty aversion, and
#             this modulation itself depends on likelihood insensitivity.
# ============================================================
print("Figure 6 …")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)

for ax, a in zip(axes, alphas):
    for b in betas:
        sub = df[(df.Alpha_Loc == a) & (df.Beta_Loc == b)]
        by_t = sub.groupby("Threshold")["Crime_Final"].agg(["mean", "std", "count"])
        mn = by_t["mean"].values
        n  = by_t["count"].values
        se = by_t["std"].values / np.sqrt(n)
        # 95% Monte Carlo CI via bootstrap per threshold
        lows, highs = [], []
        for th in threshs:
            vals = sub[sub.Threshold == th]["Crime_Final"]
            lo, hi = ci95(vals)
            lows.append(lo); highs.append(hi)
        color = BETA_C[b]
        ax.plot(threshs, mn, color=color, marker="o", ms=5,
                markeredgecolor=SURF, markeredgewidth=0.8,
                label=BETA_L[b], zorder=3, lw=1.6)
        ax.fill_between(threshs, lows, highs, color=color, alpha=0.14, linewidth=0)
    a_label = {0.5: "α = 0.5\n(Insensitive)",
               1.0: "α = 1.0 (No insensitivity)",
               2.0: "α = 2.0\n(Oversensitive)"}[a]
    tidy(ax, a_label,
         "Relative desperation threshold (ν)",
         YLAB if a == alphas[0] else None)
    ax.set_xticks(threshs)
    ax.set_xticklabels([f"{t:.2f}" for t in threshs])

axes[0].set_ylim(0, 0.4)
handles = [Line2D([0], [0], color=BETA_C[b], lw=1.8, label=BETA_L[b]) for b in betas]
axes[-1].legend(handles=handles, frameon=False, loc="upper left", fontsize=7.5)

save(fig, "fig6_three_way_interaction.png")


# ============================================================
# Supplementary Figure S1 – Weighting function (β = exp(delta) fix)
# ============================================================
print("Figure S1 (weighting function) …")

def w_prelec(beta, alpha, p, raw=False):
    d = beta if raw else math.exp(beta)
    return math.exp(-d * (-math.log(p)) ** alpha)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
ps = np.linspace(0.001, 0.999, 400)
for ax, raw, title in zip(axes, (True, False),
                          ("June 2026 code: δ = β", "Revised code: δ = exp(β)")):
    ax.plot(ps, ps, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for b in (-1, 0, 1):
        y = [w_prelec(b, 1.0, p, raw) for p in ps]
        ax.plot(ps, y, color=BETA_C[b], zorder=3, label=BETA_L[b])
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":", zorder=2)
    ax.scatter([0.8], [w_prelec(0, 1.0, 0.8, raw)], s=18, color=BETA_C[0],
               edgecolor=SURF, linewidth=1.2, zorder=5)
    tidy(ax, title, "Probability of a successful theft (p)", "Decision weight w(p)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.75)
axes[0].annotate("Weights above 1 give apprehension\na negative decision weight",
                 (0.05, 1.55), fontsize=7.5, color=INK2,
                 bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"))
axes[1].annotate("Every weight stays\nbetween 0 and 1",
                 (0.05, 1.55), fontsize=7.5, color=INK2,
                 bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.3"))
axes[0].legend(loc="lower right", handlelength=1.4,
               frameon=True, facecolor="white", edgecolor="white")
save(fig, "figS1_weighting_function.png")


print("\nAll figures written to", OUT)


# ============================================================
# Verification prints requested by figure_fixes_for_claude_code.md
# ============================================================
print("\n=== Figure 6 verification (condition means, 3 dp) ===")
def _mean(b, a, t):
    sub = df[(df.Beta_Loc == b) & (df.Alpha_Loc == a) & (df.Threshold == t)]
    return sub["Crime_Final"].mean()

print("\nβ = 1, α = 0.5 across ν:")
for t in threshs:
    print(f"  ν = {t:.2f}: {_mean(1, 0.5, t):.3f}")

print("\nβ = 1, α = 1.0 at higher ν:")
for t in (0.35, 0.5, 0.65):
    print(f"  ν = {t:.2f}: {_mean(1, 1.0, t):.3f}")

print("\nβ = −1, α = 2.0 at ν extremes:")
for t in (0.05, 0.65):
    print(f"  ν = {t:.2f}: {_mean(-1, 2.0, t):.3f}")

print("\nChange from ν = .05 to ν = .65 for all nine β × α cells:")
for b in betas:
    for a in alphas:
        d = _mean(b, a, 0.65) - _mean(b, a, 0.05)
        print(f"  β = {b:+d}, α = {a}: Δ = {d:+.3f}")
