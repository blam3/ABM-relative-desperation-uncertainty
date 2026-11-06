"""Single validation run of the revised model (see README.md).

Writes model_data.csv (per step) and agent_data.csv (per agent per step) to the
working directory. Defaults are the no-distortion cell: beta = 0, alpha = 1.
"""
from model import rel_DMAP_model

N_STEPS = 25  # 25 steps are reported in the main text; the supplement uses 50

validation_model = rel_DMAP_model(
    lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
    beta_loc=0, beta_scale=0.5, alpha_loc=1, alpha_scale=0.35,
    weighting="prelec_log_elevation", alpha_dist="lognormal",
    min_start_wealth=100, pareto_shape=3, pareto_scale=200, p=0.8,
    width=100, height=100, num_neighbors=20, income_rank_threshold=0.20,
    tau_softmax=3.0, theta_softmax=1.5, seed=20260918,
)

for step in range(N_STEPS):
    validation_model.step()
    if step % 5 == 0:
        print(f"Completed step {step}")

model_df = validation_model.datacollector.get_model_vars_dataframe()
agent_df = validation_model.datacollector.get_agent_vars_dataframe()
agent_df = agent_df.reset_index().rename(columns={"level_0": "Step", "level_1": "AgentID"})

model_df.to_csv("model_data.csv")
agent_df.to_csv("agent_data.csv")
print("Single model run completed.")
