"""

Mesa model for the DMURD model (revised 2026-09).

"""

import mesa
from mesa import Model
from mesa.space import SingleGrid
import numpy as np
from agents import individual


def crime_proportion(model):
    return sum(agent.decision for agent in model.agents) / (model.width * model.height)


def average_wealth(model):
    w = [agent.wealth for agent in model.agents]
    return float(np.mean(w)) if w else 0.0


def p90_p10(model):
    w = np.array([agent.wealth for agent in model.agents])
    return float(np.percentile(w, 90) - np.percentile(w, 10))


class rel_DMAP_model(Model):
    def __init__(self, lambd=0.2, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
                 beta_loc=0, beta_scale=0.5, alpha_loc=1, alpha_scale=0.35,
                 min_start_wealth=100, p=0.8, width=100, height=100,
                 num_neighbors=20, income_rank_threshold=0.05,
                 tau_softmax=3.0, theta_softmax=1.5,
                 weighting="prelec_log_elevation", alpha_dist="lognormal",
                 desperation_rule="rank", kappa=0.5,
                 pareto_shape=3, pareto_scale=200, seed=None):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, torus=True)
        self.lambd = lambd
        self.p = p
        self.weighting = weighting
        self.desperation_rule = desperation_rule
        for _, pos in self.grid.coord_iter():
            agent = individual(
                model=self, lambd=lambd, gamma=gamma, reward_rb=reward_rb,
                reward_rf=reward_rf, cost_rb=cost_rb, min_start_wealth=min_start_wealth,
                p=p, beta_loc=beta_loc, beta_scale=beta_scale,
                alpha_loc=alpha_loc, alpha_scale=alpha_scale,
                num_neighbors=num_neighbors, income_rank_threshold=income_rank_threshold,
                tau_softmax=tau_softmax, theta_softmax=theta_softmax,
                weighting=weighting, alpha_dist=alpha_dist,
                desperation_rule=desperation_rule, kappa=kappa,
                pareto_shape=pareto_shape, pareto_scale=pareto_scale)
            self.grid.place_agent(agent, pos)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Proportion crime": crime_proportion,
                "Average Wealth": average_wealth,
                "P90-P10": p90_p10,
            },
            agent_reporters={
                "Wealth": "wealth", "Decision": "decision", "Rank": "income_rank",
                "Desperate": "desperate_state", "Position": "pos", "Caught": "caught",
            },
        )
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
