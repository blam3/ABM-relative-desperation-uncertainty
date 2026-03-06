# DMAP Condition Mapping

**45 experimental conditions** from parameter sweep in `batch_run.py`

## Parameter Values
- **Beta_Loc**: [-1, 0, 1] (uncertainty aversion)
- **Alpha_Loc**: [0.5, 1, 2] (probability weighting)
- **Threshold**: [0.05, 0.2, 0.35, 0.5, 0.65] (desperation threshold)

## All 45 Combinations

| Condition | Beta | Alpha | Threshold | File Pattern |
|-----------|------|-------|-----------|--------------|
| 1 | -1 | 0.5 | 0.05 | `cond_1_beta_neg1_alpha_0.5_thresh_0.05.csv` |
| 2 | -1 | 0.5 | 0.2 | `cond_2_beta_neg1_alpha_0.5_thresh_0.2.csv` |
| 3 | -1 | 0.5 | 0.35 | `cond_3_beta_neg1_alpha_0.5_thresh_0.35.csv` |
| 4 | -1 | 0.5 | 0.5 | `cond_4_beta_neg1_alpha_0.5_thresh_0.5.csv` |
| 5 | -1 | 0.5 | 0.65 | `cond_5_beta_neg1_alpha_0.5_thresh_0.65.csv` |
| 6 | -1 | 1 | 0.05 | `cond_6_beta_neg1_alpha_1_thresh_0.05.csv` |
| 7 | -1 | 1 | 0.2 | `cond_7_beta_neg1_alpha_1_thresh_0.2.csv` |
| 8 | -1 | 1 | 0.35 | `cond_8_beta_neg1_alpha_1_thresh_0.35.csv` |
| 9 | -1 | 1 | 0.5 | `cond_9_beta_neg1_alpha_1_thresh_0.5.csv` |
| 10 | -1 | 1 | 0.65 | `cond_10_beta_neg1_alpha_1_thresh_0.65.csv` |
| 11 | -1 | 2 | 0.05 | `cond_11_beta_neg1_alpha_2_thresh_0.05.csv` |
| ... | ... | ... | ... | ... |
| 45 | 1 | 2 | 0.65 | `cond_45_beta_1_alpha_2_thresh_0.65.csv` |

## Generate Complete Mapping
```python
import itertools

beta_values = [-1, 0, 1]
alpha_values = [0.5, 1, 2]
threshold_values = [0.05, 0.2, 0.35, 0.5, 0.65]

conditions = list(itertools.product(beta_values, alpha_values, threshold_values))

for i, (beta, alpha, thresh) in enumerate(conditions, 1):
    safe_beta = str(beta).replace("-", "neg")
    filename = f"cond_{i}_beta_{safe_beta}_alpha_{alpha}_thresh_{thresh}.csv"
    print(f"Condition {i}: Beta={beta}, Alpha={alpha}, Threshold={thresh} → {filename}")
```

## Quick File Check
```bash
# Count generated files
ls -1 results/final_batch_run/cond_*.csv | wc -l

# List first 10 conditions
ls results/final_batch_run/cond_*.csv | head -10

# Check parameter combinations in actual files
head -n2 results/final_batch_run/cond_*.csv | grep "^0,0," | cut -d, -f10-12 | sort | uniq -c
```
