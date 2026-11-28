from mesa.discrete_space import FixedAgent
import numpy as np
from scipy.stats import bernoulli, t
import cognitive_functions as cf

# Search radius for the Moore neighborhood used in relative_desperation().
# radius=3 on a fully-occupied 100x100 torus gives (2x3+1)^2-1 = 48 candidate
# neighbors, from which num_neighbors (default 20) are randomly sampled via the
# model's seeded RNG.  The previous code passed num_neighbors as the radius,
# which covered (2x20+1)^2-1 = 1,680 cells - effectively the entire population.
_NEIGHBOR_RADIUS = 3

class individual(FixedAgent):
    """An agent that chooses between two options based on the subjective value."""
    def __init__(self, model, lambd, gamma, reward_rb, reward_rf, cost_rb, min_start_wealth, p, beta_loc, beta_scale,
                 alpha_loc, alpha_scale, num_neighbors, income_rank_threshold,
                 tau_softmax=3.0, theta_softmax=1.5):
        """ Create a new decision maker agent.
        Args:
            self: Object that stores characteristics of the agent.
            model: Reference to the model this agent belongs to.
            num_neighbors: Number of neighbors to consider for relative desperation.
            gamma: Utility parameter.
            reward_rb: Reward for rule breaking.
            reward_rf: Reward for following rules.
            cost_rb: Cost of rule breaking.
            min_start_wealth: The  minimum starting wealth of the agent.
            p: Probability of NOT being caught for rule-breaking.
            beta_loc: Mean optimism/pessimism (aversion) of the distribution.
            beta_scale: Scale of the distribution for beta.
            alpha_loc: Mean likelihood sensitivity of the distribution.
            alpha_scale: Scale of the distribution for alpha.
            lambd: Desperation severity parameter.
            alpha: Likelihood sensitivity.
            desperate_state: State of relative desperation (0 = not desperate, 1 = desperate).
            income_rank_threshold: Threshold for relative desperation based on income rank.
            tau_softmax: Satisficing margin in bounded_softmax (inertia toward rule-following).
            theta_softmax: Noise parameter in bounded_softmax (bounded rationality).
        """
        super().__init__(model)
        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p
        self.beta = np.clip(t.rvs(df=max(10, model.width*model.height-1),
                                  loc=beta_loc, scale=beta_scale), -10, 10)
        self.alpha = np.clip(t.rvs(df=max(10, model.width*model.height-1),
                                   loc=alpha_loc, scale=alpha_scale), 0.1, 10)

        self.wealth = np.clip(np.random.pareto(a=3) * 200 + min_start_wealth,
                              min_start_wealth, min_start_wealth * 100)
        self.income_rank_threshold = income_rank_threshold
        self.num_neighbors = num_neighbors
        self.tau_softmax   = tau_softmax
        self.theta_softmax = theta_softmax
        self.desperate_state = 0
        self.rel_deprivation = 0.0
        self.decision = 0
        self.caught = False
        self.income_rank = 0
        self.wealth_start = self.wealth
        self.wealth_end = self.wealth

    def relative_desperation(self):
        """Determine the relative desperation of the agent based on income rank.

        Searches a Moore neighborhood of radius _NEIGHBOR_RADIUS (48 candidates
        on a fully-occupied 100x100 torus) then draws a random sample of exactly
        num_neighbors agents to form the comparison group.  Sampling uses the
        model's seeded RNG so results are reproducible across replications.
        """
        all_neighbors = list(self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=_NEIGHBOR_RADIUS))
        if len(all_neighbors) > self.num_neighbors:
            neighbors = self.model.random.sample(all_neighbors, self.num_neighbors)
        else:
            neighbors = all_neighbors

        neighbor_wealths = [n.wealth for n in neighbors]
        num_poorer = sum(1 for w in neighbor_wealths if w <= self.wealth)
        self.income_rank = cf.cal_income_rank(num_poorer + 1, len(neighbors) + 1)
        self.desperate_state = 1 if self.income_rank <= self.income_rank_threshold else 0

        # Stark/Yitzhaki relative deprivation over the SAME local reference group:
        # mean upward gap to richer neighbors (= fraction richer x mean excess).
        # Translation-invariant in the gaps, so robust to negative (debt) wealth.
        if neighbor_wealths:
            upward_gaps = [w - self.wealth for w in neighbor_wealths if w > self.wealth]
            self.rel_deprivation = sum(upward_gaps) / len(neighbor_wealths)
        else:
            self.rel_deprivation = 0.0

    def compute_expected_utilities(self):
        """Choose an option based on subjective value. If agent is relatively desperate, use a different utility function."""
        if self.desperate_state == 1:
            if getattr(self.model, "use_refshift", False):
                # Reference-shift mechanism: reference point elevated by the felt gap.
                ref_point = self.wealth + self.lambd * self.rel_deprivation
                self.SV_rule_break = cf.SV_relative_desp_RB_refshift(
                    gamma=self.gamma, ref_point=ref_point, starting_wealth=self.wealth,
                    p=self.p, beta=self.beta, alpha=self.alpha,
                    reward_rb=self.reward_rb, cost_rb=self.cost_rb)
                self.SV_follow_rules = cf.SV_follow_rules_refshift(
                    reward_rf=self.reward_rf, starting_wealth=self.wealth,
                    gamma=self.gamma, ref_point=ref_point)
                probs = cf.bounded_softmax(self.SV_rule_break, self.SV_follow_rules,
                                           tau=self.tau_softmax, theta=self.theta_softmax)
                self.decision = int(bernoulli.rvs(probs[0]))
                return
            else:
                self.SV_rule_break = cf.SV_relative_desp_RB(
                    gamma=self.gamma, starting_wealth=self.wealth, lambd=self.lambd,
                    p=self.p, beta=self.beta, alpha=self.alpha,
                    reward_rb=self.reward_rb, cost_rb=self.cost_rb)
        else:
            self.SV_rule_break = cf.SV_rule_break(
                gamma=self.gamma, reward_rb=self.reward_rb, cost_rb=self.cost_rb, 
                p=self.p, beta=self.beta, alpha=self.alpha, starting_wealth=self.wealth)
        
        self.SV_follow_rules = cf.SV_follow_rules(
            reward_rf=self.reward_rf, starting_wealth=self.wealth, gamma=self.gamma)

        probs = cf.bounded_softmax(self.SV_rule_break, self.SV_follow_rules,
                                    tau=self.tau_softmax, theta=self.theta_softmax)
        self.decision = int(bernoulli.rvs(probs[0]))
    
    def step(self):
        self.wealth_start = self.wealth
        self.relative_desperation()
        self.compute_expected_utilities()
        
        # Update wealth based on decision
        if self.decision == 1:  # breaks rules
            if bernoulli.rvs(self.p) == 1:
                self.caught = False
                self.wealth += self.reward_rb
            else:
                self.caught = True
                self.wealth -= self.cost_rb
        else:
            self.wealth += self.reward_rf
        
        # Deduct living costs
        living_cost_rate = np.clip(np.random.normal(0.8, 0.1), 0.1, 1)
        self.wealth = self.wealth - (living_cost_rate * self.reward_rf)
        
        self.wealth_end = self.wealth
