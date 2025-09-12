from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
import numpy as np

# from .agents import decision_maker

class rel_DMAP_model(Model):
    """A model with a number of decision makers."""
    def __init__(self, width, height, lambd, gamma, reward_rb, reward_rf, cost_rb, min_start_wealth, p, 
                 beta_loc, beta_scale, alpha_loc, alpha_scale, num_neighbors, income_rank_threshold):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, torus=True)

        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.p = p

        self.desperate_state = 0  # Initialize desperate state to 0 (not desperate)
        self.decision = 0  # Initialize decision to 0 (follow rules)
        self.caught = False  # Initialize caught state to False
        self.income_rank = 0  # Initialize income rank to 0
        self.rb_choice = 0  # Initialize rule-breaking choice to 0 (follow rules)

        # Create agents and place them on the grid
        for _, pos in self.grid.coord_iter():
            # Create a new agent at the current position
            agent = individual(
                model=self, lambd=self.lambd, gamma=self.gamma, reward_rb=self.reward_rb,
                reward_rf=self.reward_rf, cost_rb=self.cost_rb, min_start_wealth=min_start_wealth,
                p=self.p, beta_loc=beta_loc, beta_scale=beta_scale, alpha_loc=alpha_loc, 
                alpha_scale=alpha_scale, num_neighbors=num_neighbors, income_rank_threshold=income_rank_threshold)
            # Add the agent to the grid at the current position
            self.grid.place_agent(agent, pos)

    
        self.datacollector = mesa.DataCollector(
            model_reporters={"Proportion crime": crime_proportion},
            agent_reporters={ "Wealth Start": lambda a: getattr(a, "wealth_start", a.wealth), "Wealth End": 
                             lambda a: getattr(a, "wealth_end", a.wealth), "income rank":"income_rank", 
                             "desperate_state": "desperate_state", "uncert_aver": "beta", "uncert_insensitive": "alpha",
                             "Rule-breaking choice": "decision", "Caught": "caught", "SV_rule_break": "SV_rule_break", 
                             "SV_follow_rules": "SV_follow_rules"},
                             tables = {"table": ["step", "agent_id", "decision", "wealth" ]  }
            )
        self.running = True
        self.datacollector.collect(self)
    
    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)
        

    
# Functions to calculate model metrics
def crime_proportion(model):
    agent_crimes = [agent.decision for agent in model.agents]
    n = model.width * model.height  # Total number of agents in the grid
    proportion = (sum(agent_crimes) ) / n
    return proportion

def gini_coefficient(model):
    """Calculate the Gini coefficient for the wealth distribution of agents in the model."""
    agent_wealths = [agent.wealth for agent in model.agents]
    n = len(agent_wealths)
    if n == 0:
        return 0  # Avoid division by zero if there are no agents
    sorted_wealths = np.sort(agent_wealths)
    cumulative_wealth = np.cumsum(sorted_wealths)
    total_wealth = cumulative_wealth[-1]
    
    # Gini coefficient formula
    gini = (n + 1 - 2 * np.sum(cumulative_wealth) / total_wealth) / n
    return gini
