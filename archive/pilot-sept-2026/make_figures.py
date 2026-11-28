"""Figures for the September 2026 pilot runs (see pilot/README.md)."""
import json, math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.environ.get("PILOT_DIR", HERE)
OUT = os.environ.get("FIG_DIR", os.path.join(HERE, "figures"))
os.makedirs(OUT, exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BETA_C = {-1: S2, 0: S1, 1: S3}
BETA_L = {-1: "β = −1 (uncertainty seeking)", 0: "β = 0 (no distortion)", 1: "β = 1 (uncertainty averse)"}

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
    if title: ax.set_title(title, loc="left", pad=8)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

def w_prelec(beta, alpha, p, raw):
    d = beta if raw else math.exp(beta)
    return math.exp(-d * (-math.log(p)) ** alpha)

read = lambda n: pd.read_csv(os.path.join(PILOT, n))
stats = json.load(open(os.path.join(PILOT, "stats.json")))

# ---- Figure 1: what the fix does to the weighting function -------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
ps = np.linspace(0.001, 0.999, 400)
for ax, raw, title in zip(axes, (True, False),
                          ("June 2026 code: δ = β", "Revised code: δ = exp(β)")):
    ax.plot(ps, ps, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for b in (-1, 0, 1):
        y = [w_prelec(b, 1.0, p, raw) for p in ps]
        ax.plot(ps, y, color=BETA_C[b], zorder=3, label=BETA_L[b])
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":", zorder=2)
    ax.scatter([0.8], [w_prelec(0, 1, 0.8, raw)], s=18, color=BETA_C[0],
               edgecolor=SURF, linewidth=1.2, zorder=5)
    tidy(ax, title, "probability of a successful theft, p", "decision weight w(p)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.75)
axes[0].annotate("weights above 1 give the apprehension\noutcome a negative weight", (0.06, 1.52),
                 fontsize=8, color=INK2)
axes[1].annotate("every weight stays\nbetween 0 and 1", (0.06, 1.52), fontsize=8, color=INK2)
axes[0].legend(frameon=False, loc="lower right", handlelength=1.4)
fig.suptitle("Figure 1. Probability weighting before and after the fix (α = 1)",
             x=0.005, ha="left", fontsize=10, y=1.04)
save(fig, "fig1_weighting_function.png")

# ---- Figure 2: the pilot reproduces the published numbers --------------------
ms, leg = stats["ms"], stats["legacy"]
pairs = ([(ms["beta"][k], leg["beta"][k], f"β = {k}") for k in ms["beta"]] +
         [(ms["alpha"][k], leg["alpha"][k], f"α = {k}") for k in ms["alpha"]] +
         [(ms["thresh_lo"], leg["thresh"]["0.05"], "ν = .05"),
          (ms["thresh_hi"], leg["thresh"]["0.65"], "ν = .65"),
          (ms["neutral"], leg["neutral"], "β0 α1 ν.20"),
          (ms["gap"], leg["gap"], "desperation gap"),
          (ms["step25"], leg["step25"], "step 25")])
fig, ax = plt.subplots(figsize=(5.0, 4.8))
lo, hi = 0.08, 0.34
ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=26, color=S1,
           edgecolor=SURF, linewidth=1.0, zorder=3)
offsets = {"β = -1": (6, -3), "β = 0": (7, -6), "β = 1": (6, 2), "α = 0.5": (7, 3),
           "α = 1.0": (-10, 9), "α = 2.0": (6, -9), "ν = .05": (6, -7), "ν = .65": (6, 1),
           "β0 α1 ν.20": (-52, -4), "desperation gap": (-16, -12), "step 25": (8, -6)}
for x, y, lab in pairs:
    ax.annotate(lab.replace("-1", "−1"), (x, y), textcoords="offset points",
                xytext=offsets.get(lab, (6, -3)), fontsize=7, color=INK2)
tidy(ax, "Figure 2. Pilot (50 × 50) vs. published results (100 × 100)",
     "manuscript, June 2026 sweep", "pilot, June settings")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
save(fig, "fig2_pilot_validation.png")

# ---- Figure 3: offending by α and β, before and after ------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
alphas = [0.5, 1.0, 2.0]
for ax, key, title in zip(axes, ("legacy", "fullfix"),
                          ("June 2026 code", "Revised code")):
    d = stats[key]
    for b in (-1, 0, 1):
        y = [d["ba"][f"{b}|{a}"] for a in alphas]
        ax.plot(range(3), y, color=BETA_C[b], marker="o", ms=4.5,
                markeredgecolor=SURF, markeredgewidth=1.0, label=BETA_L[b])
        off = {-1: (0, 8), 0: (13, -3), 1: (0, -13)}[b]
        for xi, yi in zip(range(3), y):
            ax.annotate(f"{yi:.2f}".lstrip("0"), (xi, yi), textcoords="offset points",
                        xytext=off, ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(3), ["0.5\n(insensitive)", "1.0\n(no distortion)", "2.0\n(oversensitive)"])
    tidy(ax, title, "likelihood sensitivity, α", "proportion offending at step 25")
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(0, 0.48)
axes[0].legend(frameon=False, loc="upper right", handlelength=1.4)
fig.suptitle("Figure 3. Likelihood insensitivity reverses direction once β is on the right scale",
             x=0.005, ha="left", fontsize=10, y=1.04)
save(fig, "fig3_beta_by_alpha.png")

# ---- Figure 4: trajectories --------------------------------------------------
t = read("results_traj.csv")
t = t[t.ver == "fullfix"].groupby(["thresh", "step"])[["crime", "gap", "fdesp", "desp_rate"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
for th, c in ((0.2, S1), (0.5, S2)):
    g = t[t.thresh == th]
    axes[0].plot(g.step, g.crime, color=c)
    axes[0].annotate(f"ν = {th:.2f}".replace("0.", "."), (50, g.crime.iloc[-1]),
                     textcoords="offset points", xytext=(5, -2), color=c, fontsize=8)
    axes[1].plot(g.step, g.gap, color=c)
    axes[1].annotate(f"ν = {th:.2f}".replace("0.", "."), (50, g.gap.iloc[-1]),
                     textcoords="offset points", xytext=(5, -2), color=c, fontsize=8)
tidy(axes[0], "Offending", "simulation step", "proportion offending")
tidy(axes[1], "Wealth inequality", "simulation step", "P90 − P10 wealth gap")
for ax in axes:
    ax.axvline(25, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 56)
axes[0].set_ylim(0, 0.18)
axes[0].annotate("end of the\nanalysis window", (25.8, 0.015), fontsize=7, color=INK2)
fig.suptitle("Figure 4. Offending levels off while inequality keeps growing (revised code, β = 0, α = 1)",
             x=0.005, ha="left", fontsize=10, y=1.04)
save(fig, "fig4_trajectories.png")

# ---- Figure 5: exogenous inequality -----------------------------------------
q = pd.DataFrame(stats["ineq"])
q["level"] = pd.Categorical(q["level"], ["low", "mid", "high"], ordered=True)
panels = [("rank", 0.2, "Rank rule, ν = .20"), ("rank", 0.5, "Rank rule, ν = .50"),
          ("median_ratio", 0.0, "Below half the local median")]
fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0), sharey=True)
x = np.arange(3); w = 0.36
for ax, (rule, th, title) in zip(axes, panels):
    g = q[(q["rule"] == rule) & (q["thresh"] == th)].sort_values("level")
    ax.bar(x - w / 2, g.c1, w - 0.04, color=S1, label="step 1")
    ax.bar(x + w / 2, g.cF, w - 0.04, color=S2, label="step 25")
    for xi, (a, b) in enumerate(zip(g.c1, g.cF)):
        ax.annotate(f"{a:.2f}".lstrip("0"), (xi - w / 2, a), ha="center", va="bottom",
                    fontsize=7, color=INK2)
        ax.annotate(f"{b:.2f}".lstrip("0"), (xi + w / 2, b), ha="center", va="bottom",
                    fontsize=7, color=INK2)
    ax.set_xticks(x, ["low\n(Gini ≈ .08)", "middle\n(≈ .30)", "high\n(≈ .52)"])
    tidy(ax, title, "initial wealth inequality", "proportion offending" if rule == "rank" and th == 0.2 else None)
    ax.set_ylim(0, 0.24)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Figure 5. Initial inequality raises offending at first, then the conditions converge",
             x=0.005, ha="left", fontsize=10, y=1.05)
save(fig, "fig5_exogenous_inequality.png")

# ---- Figure 6: why offending levels off -------------------------------------
u = lambda z: math.copysign(abs(z) ** 0.3, z)
wgt = stats["fullfix"]["wp"]["0"]
gaps = np.linspace(0, 200, 400)
pr = [1 / (1 + math.exp(-((wgt * u(50 - g) + (1 - wgt) * u(-250 - g) - u(25 - g) - 3) / 1.5))) for g in gaps]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(gaps, pr, color=S1)
ax.axvspan(25, 50, color=S2, alpha=0.12, lw=0)
ax.annotate("a successful theft still\ncloses the gap", (26, 0.46), fontsize=8, color=INK2)
ax.annotate("even a successful theft\nleaves the agent behind", (95, 0.20), fontsize=8, color=INK2)
tidy(ax, "Figure 6. A desperate agent's probability of offending, by the felt gap",
     "λD (weighted wealth gap to richer neighbors)", "P(offend) in one step")
ax.set_xlim(0, 200); ax.set_ylim(0, 0.55)
save(fig, "fig6_desperation_gap_curve.png")
