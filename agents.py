"""
Agent class for the DMURD model.

REVISION (2026-09)
- weighting: see cognitive_functions.prelec (default fixes beta <= 0).
- alpha_dist: "lognormal" (default) samples alpha = alpha_loc * exp(z * alpha_scale),
  which keeps alpha > 0 without piling agents at a clip boundary. "t_clip" is the
  legacy draw (t, clipped at 0.1), which placed about 21% of agents exactly at
  0.1 in the alpha_loc = 0.5 condition.
- desperation_rule: "rank" (default, income rank <= threshold) or
  "median_ratio" (wealth below kappa x median neighbor wealth). Under "rank" the
  share of desperate agents is fixed near the threshold by construction, so
  inequality cannot increase the number of desperate agents.
- pareto_shape / pareto_scale: initial wealth distribution (legacy 3 / 200).
The legacy (June 2026) model is reproduced with
weighting="prelec_raw", alpha_dist="t_clip", desperation_rule="rank".
"""
from mesa.discrete_space import FixedAgent
import numpy as np
from scipy.stats import bernoulli, t
import cognitive_functions as cf

_NEIGHBOR_RADIUS = 3


class individual(FixedAgent):
    def __init__(self, model, lambd, gamma, reward_rb, reward_rf, cost_rb, min_start_wealth, p,
                 beta_loc, beta_scale, alpha_loc, alpha_scale, num_neighbors,
                 income_rank_threshold, tau_softmax=3.0, theta_softmax=1.5,
                 weighting="prelec_log_elevation", alpha_dist="lognormal",
                 desperation_rule="rank", kappa=0.5,
                 pareto_shape=3, pareto_scale=200):
        super().__init__(model)
        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p
        self.weighting = weighting
        self.desperation_rule = desperation_rule
        self.kappa = kappa
        df = max(10, model.width * model.height - 1)  # df this large = normal distribution
        self.beta = np.clip(t.rvs(df=df, loc=beta_loc, scale=beta_scale), -10, 10)
        if alpha_dist == "t_clip":
            self.alpha = np.clip(t.rvs(df=df, loc=alpha_loc, scale=alpha_scale), 0.1, 10)
        elif alpha_dist == "lognormal":
            self.alpha = float(alpha_loc * np.exp(t.rvs(df=df, loc=0, scale=alpha_scale)))
        else:
            raise ValueError("alpha_dist must be 't_clip' or 'lognormal'")
        self.wealth = np.clip(np.random.pareto(a=pareto_shape) * pareto_scale + min_start_wealth,
                              min_start_wealth, min_start_wealth * 100)
        self.income_rank_threshold = income_rank_threshold
        self.num_neighbors = num_neighbors
        self.tau_softmax = tau_softmax
        self.theta_softmax = theta_softmax
        self.desperate_state = 0
        self.rel_deprivation = 0.0
        self.decision = 0
        self.caught = False
        self.income_rank = 0
        self.wealth_start = self.wealth
        self.wealth_end = self.wealth

    def relative_desperation(self):
        """Sample 20 of the 48 agents within a radius-3 Moore neighborhood, compute
        income rank and the Stark/Yitzhaki deprivation gap D, and classify the
        agent as desperate or not."""
        all_neighbors = list(self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=_NEIGHBOR_RADIUS))
        if len(all_neighbors) > self.num_neighbors:
            neighbors = self.model.random.sample(all_neighbors, self.num_neighbors)
        else:
            neighbors = all_neighbors
        neighbor_wealths = [n.wealth for n in neighbors]
        num_poorer = sum(1 for w in neighbor_wealths if w <= self.wealth)
        self.income_rank = cf.cal_income_rank(num_poorer + 1, len(neighbors) + 1)
        if self.desperation_rule == "rank":
            self.desperate_state = 1 if self.income_rank <= self.income_rank_threshold else 0
        elif self.desperation_rule == "median_ratio":
            med = float(np.median(neighbor_wealths)) if neighbor_wealths else self.wealth
            line = self.kappa * med if med > 0 else med
            self.desperate_state = 1 if self.wealth < line else 0
        else:
            raise ValueError("desperation_rule must be 'rank' or 'median_ratio'")
        # D = mean upward gap to the sampled neighbors (Stark & Yitzhaki)
        if neighbor_wealths:
            upward_gaps = [w - self.wealth for w in neighbor_wealths if w > self.wealth]
            self.rel_deprivation = sum(upward_gaps) / len(neighbor_wealths)
        else:
            self.rel_deprivation = 0.0

    def compute_expected_utilities(self):
        if self.desperate_state == 1:
            ref_point = self.wealth + self.lambd * self.rel_deprivation
            self.SV_rule_break = cf.SV_relative_desp_RB_refshift(
                gamma=self.gamma, ref_point=ref_point, starting_wealth=self.wealth,
                p=self.p, beta=self.beta, alpha=self.alpha,
                reward_rb=self.reward_rb, cost_rb=self.cost_rb, weighting=self.weighting)
            self.SV_follow_rules = cf.SV_follow_rules_refshift(
                reward_rf=self.reward_rf, starting_wealth=self.wealth,
                gamma=self.gamma, ref_point=ref_point)
        else:
            self.SV_rule_break = cf.SV_rule_break(
                gamma=self.gamma, reward_rb=self.reward_rb, cost_rb=self.cost_rb,
                p=self.p, beta=self.beta, alpha=self.alpha,
                starting_wealth=self.wealth, weighting=self.weighting)
            self.SV_follow_rules = cf.SV_follow_rules(
                reward_rf=self.reward_rf, starting_wealth=self.wealth, gamma=self.gamma)
        probs = cf.bounded_softmax(self.SV_rule_break, self.SV_follow_rules,
                                   tau=self.tau_softmax, theta=self.theta_softmax)
        self.decision = int(bernoulli.rvs(probs[0]))

    def step(self):
        self.wealth_start = self.wealth
        self.relative_desperation()
        self.compute_expected_utilities()
        if self.decision == 1:  # offends
            if bernoulli.rvs(self.p) == 1:
                self.caught = False
                self.wealth += self.reward_rb
            else:
                self.caught = True
                self.wealth -= self.cost_rb
        else:  # law-abiding
            self.wealth += self.reward_rf
        living_cost_rate = np.clip(np.random.normal(0.8, 0.1), 0.1, 1)
        self.wealth = self.wealth - (living_cost_rate * self.reward_rf)
        self.wealth_end = self.wealth
