import mesa
from mesa.discrete_space import FixedAgent, OrthogonalMooreGrid
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
import cognitive_functions as cf
from scipy.stats import t # For t-distribution

class individual(FixedAgent):
    """An agent that chooses between two options based on the subjective value."""
    def __init__(self, model, lambd, gamma, reward_rb, reward_rf, cost_rb, min_start_wealth, p, beta_loc, beta_scale, 
                 alpha_loc, alpha_scale, num_neighbors, income_rank_threshold):
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
            radius: Radius for neighborhood search.
        """
        super().__init__(model)
        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p
        self.beta = (t.rvs(df=(model.width*model.height)-1, loc=beta_loc, scale=beta_scale, size=1)).item()
        self.alpha = (t.rvs(df=(model.width*model.height)-1, loc=alpha_loc, scale=alpha_scale, size=1)).item()

        # TODO(fropm methods): can you say why t-dist for alpha/beta + Pareto for wealth (heterogeneity + skew).

        self.wealth = (np.random.pareto(a=3, size=1) * 200 + min_start_wealth).item() # Initial wealth drawn from a Pareto distribution
        self.income_rank_threshold = income_rank_threshold
        self.num_neighbors = num_neighbors
        self.desperate_state = 0  # Initialize desperate state to 0 (not desperate)
        self.decision = 0  # Initialize decision to 0 (follow rules)
        self.caught = False  # Initialize caught state to False
        self.income_rank = 0  # Initialize income rank to 0
        self.rb_choice = 0  # Initialize rule-breaking choice to 0 (follow rules)

    def relative_desperation(self):
        """Determine the relative desperation of the agent based on income rank."""
        # Get neighbors within the specified radius
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=self.num_neighbors)
        # TODO(from methods): i think maybe clarify how num_neighbors is a *radius*, how income_rank is computed, and why this threshold, but subject to variability

        # Get a vector of the wealth of the agents in the neighborhood
        neighbor_wealths = [neighbor.wealth for neighbor in neighbors] 
        # Count the number of neighbors with wealth lower than the agent's wealth
        num_poorer_neighbors = sum(1 for w in neighbor_wealths if w <= self.wealth)

        # Calculate the income rank of the agent in its neighborhood
        income_rank = cf.income_rank(i=num_poorer_neighbors, n=len(neighbors))
        self.income_rank = income_rank  # Store the income rank in the agent's attributes

        # Label the agent as desperate if their income rank is less than 0
        if income_rank < self.income_rank_threshold:  # Threshold for relative desperation
            self.desperate_state = 1  # Agent is relatively desperate
        else:
            self.desperate_state = 0  # Agent is not relatively desperate


    def compute_expected_utilities(self):
        """Choose an option based on subjective value. If agent is relatively desperate, use a different utility function."""
        if self.desperate_state == 1:
            # If the agent is relatively desperate, use a different utility function
            self.SV_rule_break = cf.SV_relative_desp_RB(
            gamma=self.gamma, starting_wealth=self.wealth, lambd=self.lambd, p=self.p, beta=self.beta, alpha=self.alpha, reward_rb=self.reward_rb, cost_rb=self.cost_rb)
        else:
            self.SV_rule_break = cf.SV_rule_break(gamma=self.gamma, reward_rb=self.reward_rb, cost_rb=self.cost_rb, p=self.p, beta=self.beta, alpha=self.alpha, starting_wealth=self.wealth)  # SV of rule breaking
        self.SV_follow_rules = cf.SV_follow_rules(reward_rf=self.reward_rf, starting_wealth=self.wealth, gamma=self.gamma) # SV of following rules

        # Compute the softmax probabilities
        # TODO(from methods): maybe note how/why we use softmax (stochastic choice) between SV_rb/SV_rf; p = Prob(not caught) feeding those SVs
        probs = cf.softmax(self.SV_rule_break, self.SV_follow_rules)
        
        rb_choice = bernoulli.rvs(probs[0])  # Bernoulli trial for rule breaking choice
        if rb_choice==1:
            self.decision = 1 # breaks rules
        else:
            self.decision = 0 # follows rules
        return rb_choice
    
    def decision_cycle(self):
        self.wealth_start = self.wealth
        self.relative_desperation() # Determine if the agent is relatively desperate
        self.compute_expected_utilities()
    
    def step(self):
        self.decision_cycle()

        # Log the first value
        self.model.datacollector.add_table_row(
            "table",
            {"step": self.model.steps, "agent_id": self.unique_id, "decision": self.decision, "wealth": self.wealth}
        )
        self.decision_cycle()

        # Log the second value
        self.model.datacollector.add_table_row(
            "table",
            {"step": self.model.steps, "agent_id": self.unique_id, "decision": self.decision, "wealth": self.wealth}
        )

        # Log the third value
        self.model.datacollector.add_table_row(
            "table",
            {"step": self.model.steps, "agent_id": self.unique_id, "decision": self.decision, "wealth": self.wealth}
        )

        # Log the fourth value
        self.model.datacollector.add_table_row(
            "table",
            {"step": self.model.steps, "agent_id": self.unique_id, "decision": self.decision, "wealth": self.wealth}
        )

        # Log the fifth value
        self.model.datacollector.add_table_row(
            "table",
            {"step": self.model.steps, "agent_id": self.unique_id, "decision": self.decision, "wealth": self.wealth}
        )
   
        # Update wealth based on decision
        if self.decision == 1: # breaks rules
            # If the agent chooses to break the rules and is NOT caught
            if bernoulli.rvs(self.p) == 1:
                self.caught = False
                self.wealth += self.reward_rb
            # If the agent chooses to break the rules and is caught
            else:
                self.caught = True
                self.wealth -= self.cost_rb
        else: 
            # If the agent chooses to follow the rules
            self.wealth += self.reward_rf
        # Deduct the cost of living (e.g., paying for food, rent, etc.)
        self.wealth -= np.random.lognormal(mean=2.5, sigma=0.1, size=1)[0]  # Cost of living drawn from a log-normal distribution

        # Update the agent's wealth based on the choice made
        self.wealth_end = self.wealth