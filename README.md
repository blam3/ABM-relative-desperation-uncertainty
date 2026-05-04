# Decision-Making under Poverty and Uncertainty

A growing body of literature suggests that poverty shapes not only material outcomes but the cognitive and motivational processes through which individuals evaluate risk and make decisions. This repository contains an agent-based model (ABM) that simulates how relative deprivation and uncertainty interact to influence rule-breaking behavior, drawing on Prospect Theory and the desperation threshold framework (de Courson et al., 2025).

## Model overview

Agents occupy a 100×100 toroidal grid (10,000 agents) and decide each step whether to follow rules — yielding a smaller, certain reward — or break them, accepting a higher expected payoff alongside meaningful risk of loss. Notably, agents do not evaluate risk in isolation: each step, they rank their wealth relative to neighbors, and those in the bottom fraction of their local income distribution enter a *desperate* state in which the subjective value of rule-breaking is amplified. This design reflects empirical evidence that relative deprivation, rather than absolute poverty, is more robustly associated with risk-taking and rule-breaking behavior.

Individual heterogeneity in risk perception is captured through agent-level uncertainty aversion (β) and probability sensitivity (α) parameters, each drawn from a t-distribution, consistent with the well-documented variability in probability weighting observed across individuals (Gonzalez Jimenez, 2024).

### Key parameters

| Parameter | What it controls |
|-----------|-----------------|
| `lambd` (λ) | Desperation severity — how much relative poverty amplifies rule-breaking utility |
| `beta_loc` (β) | Uncertainty aversion — how agents distort perceived probabilities via Prelec weighting |
| `alpha_loc` (α) | Probability sensitivity — curvature of the Prelec probability weighting function |
| `income_rank_threshold` | Fraction of the local neighborhood classified as desperate |

Full parameter documentation: [`docs/condition-mapping.md`](docs/condition-mapping.md)  
Output format: [`docs/data-dictionary.md`](docs/data-dictionary.md)

## Installation

```bash
pip install -r requirements.txt
```

Or minimal install:

```bash
pip install mesa numpy pandas scipy matplotlib seaborn
```

## How to run

**Interactive demo (Google Colab — no setup needed):**  
Open [`notebooks/00_colab_demo.ipynb`](notebooks/00_colab_demo.ipynb) in Colab. All model code is self-contained; no external files are required.

**Single test run (local):**

```bash
cd DMAP
python run.py
# outputs: model_data.csv, agent_data.csv, table_data.csv
```

**Full batch run (local, all 45 conditions sequentially):**

```bash
python DMAP/batch_run.py
```

**Cluster (SLURM array job — recommended for full study):**

```bash
sbatch --array=1-45 slurm_script.sh
```

`batch_run.py` auto-detects `$SLURM_ARRAY_TASK_ID` and runs only its assigned condition.

## Repository structure

```
DMAP/                    # Core model package
  model.py               # Model container, grid, data collection
  agents.py              # Agent class — decision logic, wealth updates
  cognitive_functions.py # Prospect Theory utility, Prelec weighting, softmax
  batch_run.py           # Cluster/batch execution (SLURM-aware)
  run.py                 # Single test run

notebooks/               # Analysis and demo notebooks
  00_colab_demo.ipynb    # Self-contained Colab demo (start here)
  01_main_analysis.ipynb # Primary model analysis
  02_bounded_rationality.ipynb
  03_desperation_threshold.ipynb
  04_reference_dependence.ipynb
  05_relative_desperation.ipynb
  06_test_model_aug5.ipynb
  07_test_varying_AA_aug8.ipynb

docs/                    # Documentation
  data-dictionary.md     # Output CSV column spec
  condition-mapping.md   # 45-condition parameter matrix
  analysis-pipeline.md   # Analysis templates

archive/                 # Early drafts — not part of canonical model
```

## Further reading

This model is drawn from a broader literature on decision-making under poverty and uncertainty. Key references include:

- de Courson, B., Frankenhuis, W. E., & Nettle, D. (2025). Poverty is associated with both risk avoidance and risk taking: empirical evidence for the desperation threshold model from the UK and France. *Proceedings of the Royal Society B*, 292(2040), 20242071.
- Gonzalez Jimenez, V. H. (2024). Poverty and uncertainty attitudes (No. TI 2024-058/I). Tinbergen Institute Discussion Paper.
