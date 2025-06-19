import mesa
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
import cognitive_functions as cf

class decision_maker(mesa.Agent):
    """An agent that chooses between two options based on expected utility."""
    def __init__(self, model, gamma, reward_rb, reward_rf, cost_rb, starting_wealth, p, beta, alpha):
        """ Create a new decision maker agent.
        Args:
            self: Object that stores characteristics of the agent.
            model: Reference to the model this agent belongs to.
            gamma: Utility parameter.
            reward_rb: Reward for rule breaking.
            reward_rf: Reward for following rules.
            cost_rb: Cost of rule breaking.
            starting_wealth: Wealth of the agent.
            p: Probability of being caught for rule-breaking.
            beta: Optimism/pessimism.
            alpha: Likelihood sensitivity.
        """
        super().__init__(model)
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p
        self.beta = beta
        self.alpha = alpha
        self.wealth = starting_wealth

    def compute_expected_utilities(self):
        """Choose an option based on subjective value."""
        self.EU_rule_break = cf.EU_rule_break(self.gamma, self.reward_rb, self.cost_rb, self.p, self.beta, self.alpha, self.wealth)  # EU of rule breaking
        self.EU_follow_rules = cf.EU_follow_rules(self.gamma, self.reward_rf, self.wealth) # EU of following rules
        probs = cf.softmax(self.EU_rule_break, self.EU_follow_rules)
        
        rb_choice = bernoulli.rvs(probs[0])  # Bernoulli trial for rule breaking choice
        if rb_choice==1:
            print(f"Agent {self.unique_id} chose to break the rules with probability {probs[0]}")
        else:
            print(f"Agent {self.unique_id} chose to follow the rules with probability {probs[1]}")
        return rb_choice
    
    def update_wealth(self):
        """Update the agent's wealth based on the choice made."""
        if self.compute_expected_utilities() == 1:
            # If the agent chooses to break the rules and is NOT caught
            if bernoulli.rvs(1 - self.p):
                self.wealth += self.reward_rb
            # If the agent chooses to break the rules and is caught
            else:
                self.wealth -= self.cost_rb
                print(f"Agent {self.unique_id} was caught breaking the rules and lost {self.cost_rb}. Current wealth: {self.wealth}")
        else:
            # If the agent chooses to follow the rules
            self.wealth += self.reward_rf
        print(f"Agent {self.unique_id} updated wealth: {self.wealth}")
    
    def step(self):
        self.compute_expected_utilities()  # Call the utility computation method
        self.update_wealth()
        self.wealth -= 10 # Deduct the cost of living (e.g., paying for food, rent, etc.)
    

from mesa import Model
from mesa.datacollection import DataCollector

# DMAP_model class
# This class represents a model with multiple decision makers who make choices based on expected utility.
# It initializes a number of agents and collects data on their choices.
class DMAP_model(Model):
    """A model with a number of decision makers."""
    
    def __init__(self, N, gamma, reward_rb, reward_rf, cost_rb, starting_wealth, p, beta, alpha):
        super().__init__()
        self.num_agents = N
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.wealth = starting_wealth
        self.p = p
        self.beta = beta
        self.alpha = alpha

        self.schedule = self.agents.shuffle_do(self)
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Choice": "step"}
        )

        # Create agents
        for i in range(self.num_agents):
            a = decision_maker(i, self, self.gamma, self.reward_rb, self.reward_rf,
                              self.cost_rb, self.wealth, self.p, self.beta, self.alpha)
            self.schedule.add(a)
    
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        # Activate all agents in random order
        self.agents.do("step")
        
    # Run the model for a specified number of steps
    test_model = DMAP_model(10, gamma=0.5, reward_rb=1000, reward_rf=50, cost_rb=25, wealth=50, p=0.05, beta=1.9, alpha=0.5)
test_model.step()