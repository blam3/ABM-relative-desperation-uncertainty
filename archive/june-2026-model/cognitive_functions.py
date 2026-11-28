import numpy as np

# ============================================================================
# SIGN CORRECTION (2026-06-03)
# ============================================================================
# All three subjective-value functions for rule-breaking previously used
#
#     SV_rb = w(p) * u(success) - (1 - w(p)) * u(failure)
#
# This is mathematically inconsistent with standard expected utility and
# cumulative prospect theory (Kahneman & Tversky 1979; Tversky & Kahneman
# 1992), and produces perverse comparative statics: under the minus-sign
# formulation, the partial derivative of the decision threshold with respect
# to the apprehension cost c has the wrong sign, so larger fines make
# rule-breaking *more* attractive rather than less.
#
# The correct binary-lottery expression is
#
#     SV_rb = w(p) * u(success) + (1 - w(p)) * u(failure)
#
# where u(failure) is the utility of post-apprehension wealth (which is
# negative whenever wealth drops below zero under the sign(x)*|x|^gamma
# value function). This matches the derivation in Modelv2.docx once the
# binary-lottery sign is correctly transcribed, and it is the standard
# form used throughout the criminology-economics and prospect-theory
# literatures.
#
# Three lines changed: the return statements of SV_rule_break (line ~79),
# SV_relative_desp_RB_refshift (line ~144), and SV_relative_desp_RB (line ~171).
# All comments referring to the old formula have been updated.
# ============================================================================


# Utility function of engaging in rule-breaking behavior
def utility_function(starting_wealth, gamma):
    """
    Calculate the utility of engaging in rule-breaking behavior.
    Parameters:
    starting_wealth (float): The amount of money.
    gamma (float): The risk aversion parameter.
    Returns:
    float: The utility of engaging in rule-breaking behavior.
    """

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    # Clamp extreme values to prevent overflow
    wealth = np.clip(starting_wealth, -1e6, 1e6)
    if wealth == 0:
        return 0.0

    # Use regular float64 instead of longdouble for better performance
    return np.sign(wealth) * (np.abs(wealth) ** gamma)


# Prelec's probability weighting function

def prelec(beta, alpha, p):
    """
    Prelec's probability weighting function.
    Parameters:
    beta (float): Uncertainty aversion parameter.
    alpha (float): Likelihood uncertainty parameter.
    p (float): Probability of not being caught engaging in rule-breaking behavior.
    Returns:
    float: Weighted probability.
    """
    if p == 0:
        return 0
    # Clamp to prevent extreme values
    log_p = np.clip(np.log(p), -50, 0)  # p is between 0 and 1, so log(p) <= 0
    inner = (-log_p) ** alpha
    log_val = -beta * inner

    # Prevent overflow
    log_val = np.clip(log_val, -700, 700)

    return np.exp(log_val)


# Decision-making functions for rule-breaking behavior

def SV_rule_break(reward_rb, cost_rb, starting_wealth, p, gamma, beta, alpha):
    """
    Subjective value of engaging in rule-breaking under bounded rationality.

    Standard binary-lottery expected utility with Prelec probability weighting:

        SV_rb = w(p) * u(theta + A) + (1 - w(p)) * u(theta - c)

    where u(.) is the sign(x)*|x|^gamma value function (concave on gains,
    convex on losses), w(.) is Prelec's probability weighting function, and
    (theta + A) and (theta - c) are post-decision wealth in the success and
    apprehension outcomes respectively.

    Parameters:
    reward_rb (float): Instant benefit from rule-breaking behavior (A).
    cost_rb (float): Magnitude of the fine if apprehended (c, positive number).
    starting_wealth (float): Initial wealth of the individual (theta).
    p (float): Probability of NOT being caught.
    gamma (float): Curvature of the value function.
    beta (float): Pessimism/optimism parameter in Prelec's weighting function.
    alpha (float): Likelihood-insensitivity parameter in Prelec's weighting function.

    Returns:
    float: Subjective value of rule-breaking.
    """

    # w(p) * u(theta + A) + (1 - w(p)) * u(theta - c)
    w_p = prelec(beta, alpha, p)
    u_success = utility_function(starting_wealth + reward_rb, gamma)
    u_failure = utility_function(starting_wealth - cost_rb, gamma)

    return w_p * u_success + (1 - w_p) * u_failure


# Decision-making function for following rules
def SV_follow_rules(reward_rf, starting_wealth, gamma):
    """
    Subjective value of following the rules (deterministic outcome).

    Parameters:
    reward_rf (float): Instant benefit of following the rules (B).
    starting_wealth (float): Initial wealth of the individual (theta).
    gamma (float): Curvature of the value function.

    Returns:
    float: Subjective value of rule-following = u(theta + B).
    """
    total_wealth = reward_rf + starting_wealth
    SV_following_rules = utility_function(total_wealth, gamma=gamma)

    return SV_following_rules


# Function to calculate income rank
def cal_income_rank(i, n):
    """
    Calculate the income rank of individual i in a reference group of size n.
    """
    return (i - 1) / (n - 1)


def desperation_utility_function(lambd, gamma, starting_wealth):
    """
    Proportional desperation effect based on relative wealth impact.

    Theory: Based on Relative Deprivation Theory (Runciman, 1966) and
    Loss Aversion (Kahneman & Tversky) - desperation effect should be
    proportional to baseline utility, not absolute wealth.
    """
    base_utility = utility_function(starting_wealth, gamma)

    # Desperation as percentage increase in utility
    # This prevents extreme values while maintaining meaningful differences
    desperation_multiplier = 1 + (lambd * 0.5)  # 0.5 scaling factor

    return base_utility * desperation_multiplier


# ============================================================================
# REFERENCE-SHIFT MECHANISM (relative-deprivation as an elevated reference point)
# Theory: Runciman (1966) upward comparison + Kahneman & Tversky (1979)
# reference dependence. The felt gap RD is the Stark/Yitzhaki relative-deprivation
# measure (mean upward gap to the local reference group); its population mean is
# mean*Gini (Yitzhaki 1979), so deepening inequality raises felt deprivation by
# construction. The deprived agent evaluates outcomes relative to R = wealth +
# lambd*RD rather than to zero, placing it in the loss domain in proportion to
# how far it has fallen behind. Because own wealth cancels, the choice depends on
# the GAP, not the absolute balance - which removes the wealth-accumulation drift.
# ============================================================================

def SV_relative_desp_RB_refshift(gamma, ref_point, starting_wealth, p, beta, alpha,
                                 reward_rb, cost_rb):
    """Subjective value of rule-breaking for a deprived agent, evaluated relative
    to an elevated reference point ref_point = starting_wealth + lambd*RD.

    Uses the corrected binary-lottery form:
        SV_rb = w(p) * u((theta + A) - R) + (1 - w(p)) * u((theta - c) - R)
    """
    w_p = prelec(beta, alpha, p)
    u_success = utility_function((starting_wealth + reward_rb) - ref_point, gamma)
    u_failure = utility_function((starting_wealth - cost_rb) - ref_point, gamma)
    return w_p * u_success + (1 - w_p) * u_failure


def SV_follow_rules_refshift(reward_rf, starting_wealth, gamma, ref_point):
    """Rule-following value for a deprived agent, evaluated relative to the SAME
    reference point. Both options must share the reference or the comparison is
    incoherent."""
    return utility_function((starting_wealth + reward_rf) - ref_point, gamma)


def SV_relative_desp_RB(gamma, lambd, starting_wealth, p, beta, alpha, reward_rb, cost_rb):
    """
    Subjective value of rule-breaking for a deprived agent under the multiplicative
    desperation mechanism (proportional utility boost via desperation_utility_function).

    Uses the corrected binary-lottery form:
        SV_rb = w(p) * u_desp(theta + A) + (1 - w(p)) * u_desp(theta - c)

    Parameters:
    gamma (float): Curvature of the value function.
    lambd (float): Desperation intensity.
    starting_wealth (float): Initial wealth of the individual (theta).
    p (float): Probability of NOT being caught.
    beta (float): Pessimism/optimism parameter.
    alpha (float): Likelihood-insensitivity parameter.
    reward_rb (float): Reward if successful rule-breaking (A).
    cost_rb (float): Fine if apprehended (c).

    Returns:
    float: Subjective value of rule-breaking under desperation.
    """

    w_p = prelec(beta, alpha, p)

    u_success = desperation_utility_function(lambd, gamma, starting_wealth + reward_rb)
    u_failure = desperation_utility_function(lambd, gamma, starting_wealth - cost_rb)

    return w_p * u_success + (1 - w_p) * u_failure


def bounded_softmax(SV_rb, SV_rf, tau=3, theta=1.5):
    """
    Probabilistic bounded-rational choice rule.

    Parameters
    ----------
    SV_rb : float
        Subjective value of rule-breaking.
    SV_rf : float
        Subjective value of rule-following.
    tau : float
        Satisficing threshold - the minimum margin needed for rule-breaking
        to feel 'worth it'. Larger tau = stronger inertia toward rule-following.
    theta : float
        Noise / bounded-rationality parameter.
        Larger theta = more random, less sensitive to value differences.

    Returns
    -------
    list of float
        [P(rule-break), P(rule-follow)]
    """
    delta = (SV_rb - SV_rf - tau) / theta
    # Prevent overflow in exp
    delta = np.clip(delta, -500, 500)
    p_rb = 1 / (1 + np.exp(-delta))
    return [p_rb, 1 - p_rb]
