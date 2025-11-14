import mesa
from mesa import Agent
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import cognitive_functions as cf

class individual(Agent):
    """An agent that chooses between two options based on the subjective value."""
    def __init__(self, model, lambd, gamma, reward_rb, reward_rf, cost_rb, min_start_wealth, p, beta_loc, beta_scale, 
                 alpha_loc, alpha_scale, num_neighbors, income_rank_threshold):
        """ Create a new decision maker agent.
        Args:
            model: Reference to the model this agent belongs to.
            num_neighbors: Number of neighbors to consider for relative desperation.
            gamma: Utility parameter.
            reward_rb: Reward for rule breaking.
            reward_rf: Reward for following rules.
            cost_rb: Cost of rule breaking.
            min_start_wealth: The minimum starting wealth of the agent.
            p: Probability of NOT being caught for rule-breaking.
            beta_loc: Mean optimism/pessimism (aversion) of the distribution.
            beta_scale: Scale of the distribution for beta.
            alpha_loc: Mean likelihood sensitivity of the distribution.
            alpha_scale: Scale of the distribution for alpha.
            lambd: Desperation severity parameter.
            income_rank_threshold: Threshold for relative desperation based on income rank.
        """
        super().__init__(model)
        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p
        
        # Draw beta and alpha from t-distribution
        n_agents = model.width * model.height
        self.beta = (t.rvs(df=n_agents - 1, loc=beta_loc, scale=beta_scale, size=1)).item()
        self.alpha = (t.rvs(df=n_agents - 1, loc=alpha_loc, scale=alpha_scale, size=1)).item()

        # Initialize wealth from Pareto distribution
        self.wealth = (np.random.pareto(a=3, size=1) * 200 + min_start_wealth).item()
        self.income_rank_threshold = income_rank_threshold
        self.num_neighbors = num_neighbors
        self.desperate_state = 0
        self.decision = 0
        self.caught = False
        self.income_rank = 0
        self.rb_choice = 0
        self.wealth_start = self.wealth
        self.wealth_end = self.wealth
        self.SV_rule_break = 0
        self.SV_follow_rules = 0

    def relative_desperation(self):
        """Determine the relative desperation of the agent based on income rank."""
        # Get neighbors within the specified radius
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.num_neighbors
        )
        
        # Get wealth of neighbors
        neighbor_wealths = [neighbor.wealth for neighbor in neighbors]
        
        # Count number of neighbors with wealth lower than agent's wealth
        num_poorer_neighbors = sum(1 for w in neighbor_wealths if w <= self.wealth)
        
        # Calculate income rank using cognitive function
        income_rank = cf.cal_income_rank(i=num_poorer_neighbors + 1, n=len(neighbors) + 1)
        self.income_rank = income_rank
        
        # Label agent as desperate if income rank is below threshold
        if income_rank < self.income_rank_threshold:
            self.desperate_state = 1
        else:
            self.desperate_state = 0

    def compute_expected_utilities(self):
        """Compute expected utilities and make decision based on subjective value."""
        if self.desperate_state == 1:
            # Use desperation utility function
            self.SV_rule_break = cf.SV_relative_desp_RB(
                gamma=self.gamma, 
                lambd=self.lambd, 
                starting_wealth=self.wealth, 
                p=self.p, 
                beta=self.beta, 
                alpha=self.alpha, 
                reward_rb=self.reward_rb, 
                cost_rb=self.cost_rb
            )
        else:
            # Use standard utility function
            self.SV_rule_break = cf.SV_rule_break(
                reward_rb=self.reward_rb,
                cost_rb=self.cost_rb,
                starting_wealth=self.wealth,
                p=self.p,
                gamma=self.gamma,
                beta=self.beta,
                alpha=self.alpha
            )
        
        # Calculate utility of following rules
        self.SV_follow_rules = cf.SV_follow_rules(
            reward_rf=self.reward_rf, 
            starting_wealth=self.wealth, 
            gamma=self.gamma
        )
        
        # Use bounded softmax decision rule
        tau = 1.0  # Satisficing threshold
        theta = 1.0  # Noise parameter
        probs = cf.bounded_softmax(self.SV_rule_break, self.SV_follow_rules, tau=tau, theta=theta)
        
        # Make probabilistic decision
        rb_choice = bernoulli.rvs(probs[0])
        self.decision = int(rb_choice)
        
        return rb_choice
    
    def decision_cycle(self):
        """Execute one decision cycle."""
        self.wealth_start = self.wealth
        self.relative_desperation()
        self.compute_expected_utilities()
    
    def step(self):
        """Execute one step of the agent."""
        # Run decision cycle
        self.decision_cycle()
        
        # Log data to table multiple times
        for _ in range(5):
            self.model.datacollector.add_table_row(
                "table",
                {
                    "step": self.model.steps, 
                    "agent_id": self.unique_id, 
                    "decision": self.decision, 
                    "wealth": self.wealth
                }
            )
   
        # Update wealth based on decision
        if self.decision == 1:  # Rule breaking
            if bernoulli.rvs(self.p) == 1:  # Not caught
                self.caught = False
                self.wealth += self.reward_rb
            else:  # Caught
                self.caught = True
                self.wealth -= self.cost_rb
        else:  # Following rules
            self.wealth += self.reward_rf
        
        # Deduct cost of living
        living_cost_rate = np.clip(np.random.normal(0.6, 0.1), 0.1, 1)
        self.wealth = self.wealth - (living_cost_rate * self.reward_rf)
        
        # Update end wealth
        self.wealth_end = self.wealth
