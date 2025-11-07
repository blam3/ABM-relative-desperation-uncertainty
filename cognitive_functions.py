
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
    try:
        log_val = -beta * (-np.log(p)) ** alpha
    except RuntimeWarning:
        log_val = float('-inf')
    
    # Clamping extremely large or small values
    if log_val > 700:  # np.log(np.finfo(np.float64).max)
        log_val = 700
    elif log_val < -700:  # np.log(np.finfo(np.float64).tiny)
        log_val = -700
    
    return np.longdouble(np.exp(log_val))

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
    SV_rule_breaking = (prelec(p=p, beta=beta, alpha=alpha) * utility_function(starting_wealth=reward_rb + starting_wealth, gamma=gamma)) + np.negative(- \
                        ((1 - prelec(p=p, beta=beta, alpha=alpha)) * utility_function(starting_wealth=starting_wealth - cost_rb, gamma=gamma)) )


    return SV_rule_breaking

# Decision-making function for following rules
def SV_follow_rules(reward_rf, starting_wealth, gamma):
    """
    Decision-making function to determine whether to engage in rule-breaking behavior.
    
    Parameters:
    reward_rf (float): Instant benefit of following the rules.
    starting_wealth (float): Initial wealth of the individual.
    p (float): Probability of not being caught engaging in rule-breaking behavior.
    gamma (float): Risk aversion parameter.
    
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
    Calculate the utility of engaging in rule-breaking behavior based on relative deprivation.
    
    Parameters:
    income_rank (float): The income rank of the individual in the reference group.
    lambd (float): The relative deprivation parameter.
    starting_wealth (float): The initial wealth of the individual.
    
    Returns:
    float: Adjusted utility function value.
    """
    
    return (np.longdouble(np.sign(starting_wealth) * (np.abs(starting_wealth))**(gamma))) + lambd*(np.abs(starting_wealth))

def SV_relative_desp_RB(gamma, lambd, starting_wealth, p, beta, alpha, reward_rb, cost_rb):
    """
    Adjust the utility function based on relative deprivation.
    
    Parameters:
    income_rank (float): The income rank of the individual in the reference group.
    gamma (float): The risk aversion parameter.
    
    Returns:
    float: Adjusted utility function value.
    """

    # Adjust gamma based on relative deprivation
    SV = (prelec(p=p, beta=beta, alpha=alpha) * desperation_utility_function(starting_wealth=reward_rb + starting_wealth, gamma=gamma, lambd=lambd)) + np.negative(- \
                        ((1 - prelec(p=p, beta=beta, alpha=alpha)) * desperation_utility_function(starting_wealth=starting_wealth - cost_rb, gamma=gamma, lambd=lambd))  )
    
    return SV

# Softmax function
def softmax(SV_rule_breaking, SV_following_rules):
    """Compute the softmax of vector SVs."""
    SVs = np.array([SV_rule_breaking, SV_following_rules])  # Ensure SVs is a numpy array

    # Check for NaN or inf values
    if np.isnan(SVs).any():
        raise ValueError("Input contains NaN values.")
    if np.isinf(SVs).any():
        raise ValueError("Input contains infinity values.")

    stable_SVs = SVs - SVs.max(axis=0)  # Subtract the max value for numerical stability, preventing overflow in exp
    e_SVs = np.exp(stable_SVs)  # Exponentiate the SVs

    sum_exp_SVs = e_SVs.sum(axis=0)
    # Prevent division by zero
    if sum_exp_SVs == 0:
        return np.zeros_like(SVs)
    
    return e_SVs / sum_exp_SVs

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
    p_rb = 1 / (1 + np.exp(-delta))  # logistic transform
    return [p_rb, 1 - p_rb]

# Softmax function
def softmax(SV_rule_breaking, SV_following_rules):
    """Compute the softmax of vector SVs."""
    SVs = np.array([SV_rule_breaking, SV_following_rules])  # Ensure SVs is a numpy array

    # Check for NaN or inf values
    if np.isnan(SVs).any():
        raise ValueError("Input contains NaN values.")
    if np.isinf(SVs).any():
        raise ValueError("Input contains infinity values.")

    stable_SVs = SVs - SVs.max(axis=0)  # Subtract the max value for numerical stability, preventing overflow in exp
    e_SVs = np.exp(stable_SVs)  # Exponentiate the SVs

    sum_exp_SVs = e_SVs.sum(axis=0)
    # Prevent division by zero
    if sum_exp_SVs == 0:
        return np.zeros_like(SVs)
    
    return e_SVs / sum_exp_SVs

