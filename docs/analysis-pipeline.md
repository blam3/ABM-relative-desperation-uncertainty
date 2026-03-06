# DMAP Analysis Pipeline


## Data Loading
```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load single condition
df = pd.read_csv('results/final_batch_run/cond_11_beta_neg1_alpha_2_thresh_0.05.csv')

# Load all conditions
def load_all_conditions():
    files = Path('results/final_batch_run/').glob('cond_*.csv')
    dfs = []
    for f in files:
        condition = int(f.stem.split('_')[1])
        df = pd.read_csv(f)
        df['Condition'] = condition
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
```

## Key Analyses

### 1. Desperation Trap Detection
```python
# Track agents entering/exiting desperation
agent_trajectories = df.groupby(['ReplicationID', 'AgentID']).agg({
    'Desperate': ['first', 'last', 'mean'],
    'Wealth': ['first', 'last', 'min', 'max'],
    'Decision': 'mean'  # rule-breaking rate
}).round(3)
```

### 2. Predictive Modeling
```python
from sklearn.ensemble import RandomForestClassifier

# Predict rule-breaking from agent state
features = ['Wealth', 'Rank', 'Desperate']
X = df[features]
y = df['Decision']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### 3. Parameter Effects
```python
# Compare outcomes across conditions
condition_summary = all_data.groupby(['Beta_Loc', 'Alpha_Loc', 'Threshold']).agg({
    'Decision': 'mean',  # rule-breaking rate
    'Desperate': 'mean',  # desperation rate
    'Wealth': ['mean', 'std'],  # wealth distribution
    'Caught': 'mean'  # enforcement rate
}).round(3)
```

## Visualization Templates

### Wealth Trajectories
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Sample agent trajectories
sample_agents = df[df.ReplicationID == 0].groupby('AgentID').head(1).sample(10)
trajectory_data = df[df.AgentID.isin(sample_agents.AgentID) & (df.ReplicationID == 0)]

plt.figure(figsize=(12, 6))
for agent_id in sample_agents.AgentID:
    agent_data = trajectory_data[trajectory_data.AgentID == agent_id]
    plt.plot(agent_data.Step, agent_data.Wealth, alpha=0.7, label=f'Agent {agent_id}')
plt.xlabel('Time Step')
plt.ylabel('Wealth')
plt.title('Sample Agent Wealth Trajectories')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
```

### Decision Patterns
```python
# Rule-breaking by desperation status
plt.figure(figsize=(10, 6))
decision_by_desperate = df.groupby(['Desperate', 'Step'])['Decision'].mean().reset_index()
sns.lineplot(data=decision_by_desperate, x='Step', y='Decision', hue='Desperate')
plt.ylabel('Rule-breaking Rate')
plt.title('Decision Patterns by Desperation Status Over Time')
```

## Expected Results
1. **Desperation trap**: Agents with Desperate=1 show higher rule-breaking rates
2. **Wealth spirals**: Negative wealth agents get trapped in debt cycles
3. **Parameter effects**: Higher thresholds → more desperation → more rule-breaking
4. **Time evolution**: Wealth inequality increases over simulation steps
