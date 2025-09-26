import mesa
from mesa.discrete_space import FixedAgent, OrthogonalMooreGrid #imported but unused in this file; trim imports for clarity?
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
main_model = rel_DMAP_model(lambd=0.1, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=200, 
                            beta_loc=1, beta_scale=0.5, alpha_loc=1, alpha_scale=0.5, 
                            min_start_wealth=100, p=0.8, width=100, height=100, num_neighbors=20)
for _ in range(10):
    main_model.step()
# say why 10 steps (burn-in? demo?) and how many steps you use for results in the paper

model_df = main_model.datacollector.get_model_vars_dataframe()
agent_df = main_model.datacollector.get_agent_vars_dataframe() #note if this is per-step or end-of-step; add which variables are key for analysis
table_df = test_model.datacollector.get_table_dataframe("table")

# Save the agent DataFrame to a CSV file
agent_df.to_csv("agent_data.csv", index=False)
model_df.to_csv("model_data.csv", index=False)
table_df.to_csv("table_data.csv", index=False)