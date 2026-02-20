import numpy as np
import pandas as pd

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
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rb (float): Instant benefit from rule-breaking behavior.
    cost_rb (float): Cost of engaging in rule-breaking behavior.
    starting_wealth (float): Initial wealth of the individual.
    p (float): Probability of not being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    beta (float): Likelihood sensitivity parameter for Prelec's probability weighting function.
    alpha (float): Optimism/pessimism parameter for Prelec's probability weighting function.
    
    Returns:
    str: Decision outcome.
    """
    
    # Calculate expected utility of engaging in rule-breaking behavior
    # w(p)u(θ+A)-(1-w(p))u(θ-c)
    # SV_rule_breaking = ( (prelec(p, beta, alpha)) * (utility_function(reward_rb + starting_wealth, gamma) )  ) - ( (1-prelec(p, beta, alpha)) * utility_function(starting_wealth - cost_rb, gamma) )
    
    # w(p)u(θ+A)-(1-w(p))u(θ-c)
    # Calculate utilities with numerical stability
    w_p = prelec(beta, alpha, p)
    u_success = utility_function(starting_wealth + reward_rb, gamma)
    u_failure = utility_function(starting_wealth - cost_rb, gamma)
    
    return w_p * u_success - (1 - w_p) * u_failure

# Decision-making function for following rules
def SV_follow_rules(reward_rf, starting_wealth, gamma):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rf (float): Instant benefit of following the rules.
    starting_wealth (float): Initial wealth of the individual.
    p (float): Probability of not being caught engaging in rule-breaking behavior.
    gamma (float):Utility (commonly referred to as risk aversion) parameter.
    
    Returns:
    float: Expected utility.
    """
    
    # Calculate expected utility of not engaging in rule-breaking behavior
    total_wealth = reward_rf + starting_wealth
    SV_following_rules = utility_function(total_wealth, gamma=gamma)
    
    return SV_following_rules

# Function to calculate income rank
def cal_income_rank(i,n):
    """
    Calculate the income rank of individual i in a reference group of size n.
    """
    return (i - 1) / (n - 1)

def desperation_utility_function(lambd, gamma, starting_wealth):
    """
    Proportional desperation effect based on relative wealth impact.
    
    Theory: Based on Relative Deprivation Theory (Runciman, 1966) and 
    Loss Aversion (Kahneman & Tversky) - desperation effect should be 
    proportional to baseline utility, not absolute wealth
    """
    base_utility = utility_function(starting_wealth, gamma)
    
    # Desperation as percentage increase in utility 
    # This prevents extreme values while maintaining meaningful differences
    desperation_multiplier = 1 + (lambd * 0.5)  # 0.5 scaling factor
    
    return base_utility * desperation_multiplier

def SV_relative_desp_RB(gamma, lambd, starting_wealth, p, beta, alpha, reward_rb, cost_rb):
    """
    Adjust the utility function based on relative deprivation.
    
    Parameters:
    income_rank (float): The income rank of the individual in the reference group.
    gamma (float): The risk aversion parameter.
    
    Returns:
    float: Adjusted utility function value.
    """

    w_p = prelec(beta, alpha, p)
    
    u_success = desperation_utility_function(lambd, gamma, starting_wealth + reward_rb)
    u_failure = desperation_utility_function(lambd, gamma, starting_wealth - cost_rb)
    
    return w_p * u_success - (1 - w_p) * u_failure

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
        Satisficing threshold — the minimum margin needed for rule-breaking
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

