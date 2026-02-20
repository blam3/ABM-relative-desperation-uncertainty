import mesa
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
import numpy as np
from agents import individual

# --- GINI COEFFICIENT (Handles Negative Wealth) ---
def gini_coefficient_with_negatives(model):
    """
    Calculates the Gini coefficient using the Chen (1982) formula, 
    which is robust to negative values (debt).
    """
    # Create list from AgentSet
    wealths = np.array([a.wealth for a in model.agents])
    n = len(wealths)
    if n == 0: return 0

    # Sort wealth for the calculation
    wealths_sorted = np.sort(wealths)
    
    # Numerator: Sum of absolute differences (Gini mean difference)
    # Calculated efficiently using dot product
    index = np.arange(1, n + 1)
    numerator = 2 * np.sum((2 * index - n - 1) * wealths_sorted)
    
    # Denominator: n^2 * mean(abs(x))
    # Chen's correction normalizes by the sum of absolute values
    denominator = n * np.sum(np.abs(wealths))
    
    if denominator == 0: return 0
    
    return numerator / denominator

def crime_proportion(model):
    agent_crimes = [agent.decision for agent in model.agents]
    n = model.width * model.height 
    return sum(agent_crimes) / n

def average_wealth(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    return np.mean(agent_wealths) if len(agent_wealths) > 0 else 0

class rel_DMAP_model(Model):
    def __init__(self, lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250, 
                 beta_loc=1, beta_scale=0.5, alpha_loc=1, alpha_scale=0.5, 
                 min_start_wealth=100, p=0.8, width=100, height=100, 
                 num_neighbors=20, income_rank_threshold=0.05, seed=None):
        
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, torus=True)
        self.agents.shuffle_do("step")

        
        # Save params
        self.lambd = lambd
        self.gamma = gamma
        self.reward_rb = reward_rb
        self.reward_rf = reward_rf
        self.cost_rb = cost_rb
        self.min_start_wealth = min_start_wealth
        self.p = p
        self.beta_loc = beta_loc
        self.beta_scale = beta_scale
        self.alpha_loc = alpha_loc
        self.alpha_scale = alpha_scale
        self.num_neighbors = num_neighbors
        self.income_rank_threshold = income_rank_threshold

        # Create agents
        for _, pos in self.grid.coord_iter():
            agent = individual(
                model=self, lambd=lambd, gamma=gamma, reward_rb=reward_rb,
                reward_rf=reward_rf, cost_rb=cost_rb, min_start_wealth=min_start_wealth,
                p=p, beta_loc=beta_loc, beta_scale=beta_scale, 
                alpha_loc=alpha_loc, alpha_scale=alpha_scale, 
                num_neighbors=num_neighbors, income_rank_threshold=income_rank_threshold)
            self.grid.place_agent(agent, pos)

        # DATA COLLECTOR 
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Proportion crime": crime_proportion,
                "Gini Coefficient": gini_coefficient_with_negatives,
                "Average Wealth": average_wealth
            },
            agent_reporters={
                "Wealth": "wealth",
                "Decision": "decision",
                "Rank": "income_rank",
                "Desperate": "desperate_state",
                "Position": "pos", 
                "Caught": "caught"
            }
        )
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.agents.do("step")
        self.datacollector.collect(self)
