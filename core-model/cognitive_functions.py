"""
Cognitive functions for the relative desperation and uncertainty (DMURD) model.

REVISION (2026-09): probability weighting parameterization
-----------------------------------------------------------
The June 2026 sweep used w(p) = exp(-beta * (-ln p)^alpha) with beta entered
directly as Prelec's elevation parameter (delta). Prelec's function is only a
valid weighting function for delta > 0:
    beta = 1  -> no elevation distortion (w(p) = p when alpha = 1)
    beta = 0  -> w(p) = 1 for every p (success treated as certain)
    beta < 0  -> w(p) > 1, so the apprehension outcome receives a NEGATIVE weight
The revised default enters beta on the log scale, delta = exp(beta), so that
    beta = 0  -> delta = 1 (no elevation distortion)
    beta > 0  -> delta > 1 (pessimism; uncertainty aversion)
    beta < 0  -> delta < 1 (optimism; uncertainty seeking)
and w(p) stays inside (0, 1) for every real beta. Set
weighting="prelec_raw" to reproduce the June 2026 results exactly.

Terminology: function names keep the legacy "rule_break" / "follow_rules"
labels so existing analysis pipelines do not break. In the manuscript these
are "offending" and "law-abiding behavior".
"""
import numpy as np

WEIGHTING_OPTIONS = ("prelec_log_elevation", "prelec_raw")


def utility_function(starting_wealth, gamma):
    """Power value function u(x) = sign(x) * |x|^gamma (defined for debt)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    wealth = np.clip(starting_wealth, -1e6, 1e6)
    if wealth == 0:
        return 0.0
    return np.sign(wealth) * (np.abs(wealth) ** gamma)


def prelec(beta, alpha, p, weighting="prelec_log_elevation"):
    """
    Two-parameter Prelec (1998) probability weighting function,
        w(p) = exp(-delta * (-ln p)^alpha).

    beta  : uncertainty aversion. Under the default "prelec_log_elevation",
            delta = exp(beta): 0 = no distortion, > 0 = pessimism (aversion),
            < 0 = optimism (seeking). Under the legacy "prelec_raw",
            delta = beta (valid only for beta > 0).
    alpha : likelihood SENSITIVITY (must be > 0). alpha = 1 = no curvature;
            alpha < 1 = inverse-S shape (likelihood insensitivity: unlikely
            outcomes overweighted, likely outcomes underweighted);
            alpha > 1 = S shape (oversensitivity).
    p     : probability of NOT being apprehended.
    """
    if weighting not in WEIGHTING_OPTIONS:
        raise ValueError(f"weighting must be one of {WEIGHTING_OPTIONS}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    delta = np.exp(beta) if weighting == "prelec_log_elevation" else beta
    inner = (-np.log(p)) ** alpha
    log_val = np.clip(-delta * inner, -700, 700)
    return float(np.exp(log_val))


def _binary_lottery(w_p, u_success, u_failure):
    """Rank-dependent value of a two-outcome prospect: w(p) on the better
    outcome (success) and 1 - w(p) on the worse outcome (apprehension)."""
    return w_p * u_success + (1 - w_p) * u_failure


def SV_rule_break(reward_rb, cost_rb, starting_wealth, p, gamma, beta, alpha,
                  weighting="prelec_log_elevation"):
    """Subjective value of offending for a non-desperate agent:
        SV_off = w(p) u(theta + R1) + (1 - w(p)) u(theta - C)."""
    w_p = prelec(beta, alpha, p, weighting)
    u_success = utility_function(starting_wealth + reward_rb, gamma)
    u_failure = utility_function(starting_wealth - cost_rb, gamma)
    return _binary_lottery(w_p, u_success, u_failure)


def SV_follow_rules(reward_rf, starting_wealth, gamma):
    """Subjective value of law-abiding behavior: SV_law = u(theta + R0)."""
    return utility_function(reward_rf + starting_wealth, gamma=gamma)


def cal_income_rank(i, n):
    """Income rank of individual i in a reference group of size n (0 = poorest)."""
    return (i - 1) / (n - 1)


def SV_relative_desp_RB_refshift(gamma, ref_point, starting_wealth, p, beta, alpha,
                                 reward_rb, cost_rb, weighting="prelec_log_elevation"):
    """Subjective value of offending for a desperate agent, evaluated relative to
    the reference point R = theta + lambda * D (D = mean upward wealth gap)."""
    w_p = prelec(beta, alpha, p, weighting)
    u_success = utility_function((starting_wealth + reward_rb) - ref_point, gamma)
    u_failure = utility_function((starting_wealth - cost_rb) - ref_point, gamma)
    return _binary_lottery(w_p, u_success, u_failure)


def SV_follow_rules_refshift(reward_rf, starting_wealth, gamma, ref_point):
    """Law-abiding value for a desperate agent, relative to the same reference point."""
    return utility_function((starting_wealth + reward_rf) - ref_point, gamma)


def bounded_softmax(SV_rb, SV_rf, tau=3, theta=1.5):
    """P(offend) = logistic((SV_off - SV_law - tau) / theta).
    tau = satisficing margin; theta = decision noise. Returns [P(offend), P(law)]."""
    delta = np.clip((SV_rb - SV_rf - tau) / theta, -500, 500)
    p_rb = 1 / (1 + np.exp(-delta))
    return [p_rb, 1 - p_rb]
