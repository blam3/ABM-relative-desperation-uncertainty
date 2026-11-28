from model import rel_DMAP_model

# Run the model for a number of steps
validation_model = rel_DMAP_model(lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250, 
                            beta_loc=1, beta_scale=0.5, alpha_loc=1, alpha_scale=0.5, 
                            min_start_wealth=100, p=0.8, width=100, height=100, num_neighbors=20, income_rank_threshold=0.05)
for _ in range(100): # 25 steps will be reported in the main text, 100 for the supplemental
    validation_model.step()
    # display progress
    if _ % 10 == 0:
        print(f"Completed step {_}")

model_df = validation_model.datacollector.get_model_vars_dataframe()
agent_df = validation_model.datacollector.get_agent_vars_dataframe()
# Make "step" a column/variable name in agent_df
agent_df = agent_df.reset_index().rename(columns={"level_0": "Step", "level_1": "AgentID"})

# Save as CSV files
model_df.to_csv("model_data.csv")
agent_df.to_csv("agent_data.csv")

print("Single model run completed.")
