
import numpy as np
import pandas as pd

# Utility function of engaging in rule-breaking behavior
def utility_function(x_ij, gamma_i):
    """
    Calculate the utility of engaging in rule-breaking behavior.
    Parameters:
    x_ij (float): The amount of money.
    gamma_i (float): The risk aversion parameter.
    Returns:
    float: The utility of engaging in rule-breaking behavior.
    """
    
    if gamma_i <= 0:
        raise ValueError("gamma_i must be positive")
    
    return np.sign(x_ij) * (np.abs(x_ij))**(1 - gamma_i)

# Prelec's probability weighting function

def prelec(p_ij, beta_i, alpha_i):
    """
    Prelec's (1998) probability weighting function.
    
    Parameters:
    p_ij (ixj array): Probability value in the range [0, 1].
    alpha_i (float): Parameter that determines the curvature of the weighting function.
    
    Returns:
    float: Weighted probability.
    """
    if p_ij < 0 or p_ij > 1:
        raise ValueError("x must be in the range [0, 1]")
    
    if alpha_i <= 0:
        raise ValueError("alpha must be positive")
    
    return np.exp(-beta_i*(-np.log(p_ij))**alpha_i)

# Decision-making functions for rule-breaking behavior

def EU_rule_break(reward_rb, cost_rb, theta, p_ij, gamma_i, beta_i, alpha_i):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rb (float): Instant benefit from rule-breaking behavior.
    cost_rb (float): Cost of engaging in rule-breaking behavior.
    theta (float): Initial wealth of the individual.
    p_ij (float): Probability of being caught engaging in rule-breaking behavior.
    gamma_i (float): Risk aversion parameter.
    beta_i (float): Likelihood sensitivity parameter for Prelec's probability weighting function.
    alpha_i (float): Optimism/pessimism parameter for Prelec's probability weighting function.
    
    Returns:
    str: Decision outcome.
    """
    
    # Calculate expected utility of engaging in rule-breaking behavior
    # w(p)u(θ+A)-(1-w(p))u(θ-c)
    EU_rule_breaking = ( (prelec(p_ij, beta_i, alpha_i)) * (utility_function(reward_rb + theta, gamma_i) )  ) - ( (1-prelec(p_ij, beta_i, alpha_i)) * utility_function(theta - cost_rb, gamma_i) )
    
    return EU_rule_breaking

# Decision-making function for following rules
def EU_follow_rules(reward_rf, theta, gamma_i):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rf (float): Instant benefit of following the rules.
    theta (float): Initial wealth of the individual.
    p_ij (float): Probability of being caught engaging in rule-breaking behavior.
    gamma_i (float): Risk aversion parameter.
    
    Returns:
    float: Expected utility.
    """
    
    # Calculate expected utility of not engaging in rule-breaking behavior
    EU_following_rules = utility_function(reward_rf + theta, gamma_i)
    
    return EU_following_rules


# Softmax function
def softmax(EU_rule_breaking, EU_following_rules):
    """Compute the softmax of vector EUs."""
    EUs = np.array(EU_rule_breaking, EU_following_rules)  # Ensure EUs is a numpy array
    e_EUs = np.exp(EUs)  # Exponentiate the EUs
    return e_EUs / e_EUs.sum(axis=0)

