
# TO DO: Change EU terminology to "value".

import numpy as np
import pandas as pd

# Utility function of engaging in rule-breaking behavior
def utility_function(wealth, gamma):
    """
    Calculate the utility of engaging in rule-breaking behavior.
    Parameters:
    x (float): The amount of money.
    gamma (float): The risk aversion parameter.
    Returns:
    float: The utility of engaging in rule-breaking behavior.
    """
# keep gamma in exponent (instead of gamma - 1)
# Keep restriction that gamma must be greater than 0
    
    if gamma <= 0:
        raise ValueError("gamma_i must be positive")
    
    return (np.abs(wealth))**(gamma)

# Prelec's probability weighting function

def prelec(p, beta, alpha):
    """
    Prelec's (1998) probability weighting function.
    Parameters:
    p (ixj array): Probability value in the range [0, 1].
    alpha (float): Parameter that determines the curvature of the weighting function.
    
    Returns:
    float: Weighted probability.
    """
    if p < 0 or p > 1:
        raise ValueError("x must be in the range [0, 1]")
    
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    
    return np.exp(-beta*(-np.log(p))**alpha)

# Decision-making functions for rule-breaking behavior

def EU_rule_break(reward_rb, cost_rb, wealth, p, gamma, beta, alpha):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rb (float): Instant benefit from rule-breaking behavior.
    cost_rb (float): Cost of engaging in rule-breaking behavior.
    wealth (float): Initial wealth of the individual.
    p (float): Probability of being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    beta (float): Likelihood sensitivity parameter for Prelec's probability weighting function.
    alpha (float): Optimism/pessimism parameter for Prelec's probability weighting function.
    
    Returns:
    str: Decision outcome.
    """
    
    # Calculate expected utility of engaging in rule-breaking behavior
    # w(p)u(θ+A)-(1-w(p))u(θ-c)
    EU_rule_breaking = ( (prelec(p, beta, alpha)) * (utility_function(reward_rb + wealth, gamma) )  ) - ( (1-prelec(p, beta, alpha)) * utility_function(wealth - cost_rb, gamma) )
    
    return EU_rule_breaking

# Decision-making function for following rules
def EU_follow_rules(reward_rf, wealth, gamma):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rf (float): Instant benefit of following the rules.
    wealth (float): Initial wealth of the individual.
    p (float): Probability of being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    
    Returns:
    float: Expected utility.
    """
    
    # Calculate expected utility of not engaging in rule-breaking behavior
    EU_following_rules = utility_function(reward_rf + wealth, gamma)
    
    return EU_following_rules


# Softmax function
def softmax(EU_rule_breaking, EU_following_rules):
    """Compute the softmax of vector EUs."""
    EUs = np.array([EU_rule_breaking, EU_following_rules])  # Ensure EUs is a numpy array
    e_EUs = np.exp(EUs)  # Exponentiate the EUs
    return e_EUs / e_EUs.sum(axis=0)

