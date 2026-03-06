# DMAP Model Data Dictionary

## Quick Overview
45 CSV files from simulation studying poverty and rule-breaking. Each file = different experimental condition testing how desperation thresholds affect agent behavior.

## File Structure
`results/final_batch_run/cond_{N}_beta_{B}_alpha_{A}_thresh_{T}.csv`

**Example**: `cond_11_beta_neg1_alpha_2_thresh_0.05.csv`
- Condition 11: Beta=-1 (risk-seeking), Alpha=2, Threshold=5%

## Variables (12 columns)

| Variable | Type | Description |
|----------|------|-------------|
| ReplicationID | 0-99 | Simulation run number |
| Step | 0-24 | Time step (25 total) |
| AgentID | 1-10000 | Agent identifier |
| Wealth | Float | Agent's money (**can be negative**) |
| Decision | 0/1 | 0=follow rules, 1=break rules |
| Rank | 0-1 | Wealth percentile in neighborhood |
| Desperate | 0/1 | Below desperation threshold |
| Position | "(x,y)" | Grid coordinates |
| Caught | 0/1 | Rule-breaker caught and penalized |
| Beta_Loc | -1,0,1 | Uncertainty aversion (experiment parameter) |
| Alpha_Loc | 0.5,1,2 | Probability weighting (experiment parameter) |
| Threshold | 0.05-0.65 | Desperation threshold (experiment parameter) |

## Experimental Design
- **45 conditions**: 3 Beta × 3 Alpha × 5 Threshold combinations
- **100 replications** per condition
- **25 time steps** per simulation
- **10,000 agents** per simulation (100×100 grid)

## Key Mechanics
1. Agents choose: follow rules (+25) or break rules (+50, risk -250)
2. Bottom X% (threshold) become "desperate" → different utility function
3. Living costs deducted each step (~20)
4. Wealth can go negative (debt/poverty)

## Research Questions
- Do desperate agents get trapped in rule-breaking cycles?
- What wealth level creates "point of no return"?
- How do Beta/Alpha parameters affect outcomes?

## Data Checks
```python
# Verify file structure
df = pd.read_csv('results/final_batch_run/cond_11_beta_neg1_alpha_2_thresh_0.05.csv')
assert df.shape[1] == 12  # 12 columns
assert df.Step.max() <= 24  # 25 time steps (0-24)
assert df.AgentID.nunique() <= 10000  # Up to 10k agents
```

## Analysis Pipeline
1. **Load condition files** → combine for parameter comparisons
2. **Track trajectories** → individual agent wealth over time
3. **Model decisions** → predict rule-breaking from wealth/rank/desperation
4. **Parameter effects** → compare outcomes across Beta/Alpha/Threshold

---
See `batch_run.py` and `model.py` for implementation details.
