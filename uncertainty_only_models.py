## Gonzalez-Jimenez Model of Decision-making under uncertainty

### Probability Weighting Functions

def prob_weighting(p_ij):
    w_p = exp(-beta_i * -log(p_ij)^alpha)