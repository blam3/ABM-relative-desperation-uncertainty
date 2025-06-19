
# TO DO: Change EU terminology to "value".

# To represent ambiguity attitudes, draw values that are more extreme (e.g., drawing from a norm dist with more kurtosis)
# Could find distribution of risk and ambiguity attitudes that were found empirically.

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
# keep gamma in exponent (instead of gamma - 1)
# Keep restriction that gamma must be greater than 0
    
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    
    return np.longdouble(np.sign(starting_wealth) * (np.abs(starting_wealth))**(gamma)) 

# Prelec's probability weighting function

def prelec(p, beta, alpha):
    """
    Prelec's (1998) probability weighting function.
    Parameters:
    p (float): Probability value in the range [0, 1].
    alpha (float): Parameter that determines the curvature of the weighting function.
    
    Returns:
    float: Weighted probability.
    """
    if p < 0 or p > 1:
        raise ValueError("p must be in the range [0, 1]")
    
#    if alpha <= 0:
#        raise ValueError("alpha must be positive")
    
    return np.longdouble(np.exp(-beta*(-np.log(p))**alpha))

# Decision-making functions for rule-breaking behavior

def SV_rule_break(reward_rb, cost_rb, starting_wealth, p, gamma, beta, alpha):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rb (float): Instant benefit from rule-breaking behavior.
    cost_rb (float): Cost of engaging in rule-breaking behavior.
    starting_wealth (float): Initial wealth of the individual.
    p (float): Probability of being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    beta (float): Likelihood sensitivity parameter for Prelec's probability weighting function.
    alpha (float): Optimism/pessimism parameter for Prelec's probability weighting function.
    
    Returns:
    str: Decision outcome.
    """
    
    # Calculate expected utility of engaging in rule-breaking behavior
    # w(p)u(θ+A)-(1-w(p))u(θ-c)
    SV_rule_breaking = ( (prelec(p, beta, alpha)) * (utility_function(reward_rb + starting_wealth, gamma) )  ) - ( (1-prelec(p, beta, alpha)) * utility_function(starting_wealth - cost_rb, gamma) )
    
    return SV_rule_breaking

# Decision-making function for following rules
def SV_follow_rules(reward_rf, starting_wealth, gamma):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rf (float): Instant benefit of following the rules.
    starting_wealth (float): Initial wealth of the individual.
    p (float): Probability of being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    
    Returns:
    float: Expected utility.
    """
    
    # Calculate expected utility of not engaging in rule-breaking behavior
    total_wealth = reward_rf + starting_wealth
    SV_following_rules = utility_function(total_wealth, gamma)
    
    return SV_following_rules


# Softmax function
def softmax(SV_rule_breaking, SV_following_rules):
    """Compute the softmax of vector SVs."""
    SVs = np.array([SV_rule_breaking, SV_following_rules])  # Ensure SVs is a numpy array
    stable_SVs = SVs - SVs.max(axis=0)  # Subtract the max value for numerical stability, preventing overflow in exp
    e_SVs = np.exp(stable_SVs)  # Exponentiate the SVs
    return e_SVs / e_SVs.sum(axis=0)

