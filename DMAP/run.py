import mesa
from mesa.discrete_space import FixedAgent, OrthogonalMooreGrid
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
import cognitive_functions as cf
from scipy.stats import t # For t-distribution
from dmap.model import rel_DMAP_model
from dmap.agents import individual
from mesa.batchrunner import batch_run
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

# Run the model for a number of steps
main_model = rel_DMAP_model(lambd=1, gamma=0.3, reward_rb=50, reward_rf=10, cost_rb=200, 
                            beta_loc=1, beta_scale=1, alpha_loc=-0.5, alpha_scale=1, 
                            min_start_wealth=20, p=0.1, width=100, height=100, num_neighbors=20)
for _ in range(500):
    main_model.step()

model_df = main_model.datacollector.get_model_vars_dataframe()
agent_df = main_model.datacollector.get_agent_vars_dataframe()

# Save the agent DataFrame to a CSV file
agent_df.to_csv("agent_data.csv", index=False)
model_df.to_csv("model_data.csv", index=False)