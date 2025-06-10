
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

