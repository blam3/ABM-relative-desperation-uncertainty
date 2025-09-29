# Decision-making under Ambiguity and Poverty (DMAP) Model 

## Summary

This is the same Decision-making under Ambiguity and Poverty (DMAP) Model.

A model of agents deciding how to accumulate wealth. All agents start with varying amounts of money. Every step, each agent must decide between two options:

1) To receive a smaller certain amount of money from following the rules.

2) To break the rules (e.g., property theft) to receive a larger sum of money but with the possibility of being caught and losing money.

As the model runs, the distribution of wealth among agents is updated over time.

## Installation
To install the dependencies use pip to install mesa[rec]

    $ pip install mesa[rec]

## Technical Details

### MESA Framework
This model uses the MESA (Multi-Agent Simulation Environment) framework for agent-based modeling. MESA provides the core infrastructure for:
- Agent creation and management (`mesa.Agent`)
- Spatial environment with grid-based interactions (`mesa.space.SingleGrid`) 
- Data collection and analysis (`mesa.DataCollector`)
- Batch running for parameter sweeps (`mesa.batch_run`)

### Decision-Making: Softmax Function
Agent decisions between rule-breaking and rule-following use a softmax function (see `cognitive_functions.py`) that converts subjective values into choice probabilities. This stochastic approach models realistic decision-making where agents don't always choose the option with highest expected value, accounting for cognitive noise and imperfect rationality.
    
## How to Run
To run the model interactively, run solara run in this directory. e.g.

    $ dmap run app.py
    
Then open your browser to http://localhost:insert_id/ and press Reset, then Run.

## Files
model.py: Contains creation of agents, the environment, and management of agent execution.
agents.py: Contains logic obtaining and losing wealth.
app.py: Contains the code for the interactive DMAP visualization.

**Note**: Might be helpful to create a separate methods folder to add separate things

## Further Reading

This model is drawn from economic models of decision-making and presents an integrative approach to rule-breaking. Some examples of further reading on the topic can be found at:

Gonzalez Jimenez, V. H. (2024). Poverty and uncertainty attitudes (No. TI 2024-058/I). Tinbergen Institute Discussion Paper.

de Courson, B., Frankenhuis, W. E., & Nettle, D. (2025). Poverty is associated with both risk avoidance and risk taking: empirical evidence for the desperation threshold model from the UK and France. Proceedings B, 292(2040), 20242071.

