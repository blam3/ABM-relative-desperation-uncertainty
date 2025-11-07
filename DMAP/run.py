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
test_model = rel_DMAP_model(lambd=0.1, gamma=0.3, reward_rb=50, reward_rf=30, cost_rb=250, 
                            beta_loc=1, beta_scale=0.5, alpha_loc=1, alpha_scale=0.5, 
                            min_start_wealth=100, p=0.8, width=40, height=40, num_neighbors=20, income_rank_threshold=0)
for _ in range(30):
    test_model.step()

model_df = test_model.datacollector.get_model_vars_dataframe()
agent_df = test_model.datacollector.get_agent_vars_dataframe()
table_df = test_model.datacollector.get_table_dataframe("table")
table_df = table_df.reset_index()

# Save as CSV files
model_df.to_csv("model_data.csv")
agent_df.to_csv("agent_data.csv")
table_df.to_csv("table_data.csv")

print("Single model run completed.")
