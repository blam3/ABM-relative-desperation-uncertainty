"""
Mesa model for the DMURD model (revised 2026-09).

The multiplicative desperation mechanism (use_refshift=False) was removed:
the June 2026 sweep and the manuscript both use the reference-shift mechanism.
New options are passed through to agents (see agents.py).

Victimization (2026-11-28)
- victim_mode: "none" (default; reproduces jasss-v1: an undetected offense adds R1
  from nothing), "global" (victim drawn uniformly from all other agents) or "local"
  (victim drawn uniformly from the offender's comparison sample this step). The
  offender gains min(R1, max(victim wealth, 0)) and the victim loses the same, so
  theft is zero-sum. A detected offender pays C and no victim is drawn.
- sync: False (default; asynchronous under shuffle_do, the victim's wealth changes
  immediately) or True (all decisions and draws are made from start-of-step wealth,
  then thefts are applied in shuffled order, each capped at the victim's running
  wealth, then each agent's income, penalty and living cost).
- Both are read from DMURD_VICTIM and DMURD_SYNC (=1) when not passed explicitly.
- Victim draws use a separate generator (victim_rng), so the global NumPy stream and
  model.random see exactly the jasss-v1 draws in the same order.
"""
import os
import mesa
from mesa import Model
from mesa.space import SingleGrid
import numpy as np
from agents import individual

VICTIM_MODES = ("none", "global", "local")
_VICTIM_STREAM = 0x5649  # spawn key for the victim RNG


def resolve_victim_options(victim_mode=None, sync=None):
    """Explicit arguments win; otherwise DMURD_VICTIM / DMURD_SYNC; defaults none / False."""
    if victim_mode is None:
        victim_mode = os.environ.get("DMURD_VICTIM", "none")
    victim_mode = str(victim_mode).strip().lower()
    if victim_mode not in VICTIM_MODES:
        raise ValueError(f"victim_mode must be one of {VICTIM_MODES}, got {victim_mode!r}")
    if sync is None:
        raw = os.environ.get("DMURD_SYNC", "0").strip()
        if raw not in ("0", "1", ""):
            raise ValueError(f"DMURD_SYNC must be 0 or 1, got {raw!r}")
        sync = raw == "1"
    return victim_mode, bool(sync)


def crime_proportion(model):
    return sum(agent.decision for agent in model.agents) / (model.width * model.height)


def average_wealth(model):
    w = [agent.wealth for agent in model.agents]
    return float(np.mean(w)) if w else 0.0


def p90_p10(model):
    w = np.array([agent.wealth for agent in model.agents])
    return float(np.percentile(w, 90) - np.percentile(w, 10))


class rel_DMAP_model(Model):
    def __init__(self, lambd=0.5, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
                 beta_loc=0, beta_scale=0.5, alpha_loc=1, alpha_scale=0.35,
                 min_start_wealth=100, p=0.8, width=100, height=100,
                 num_neighbors=20, income_rank_threshold=0.05,
                 tau_softmax=3.0, theta_softmax=1.5,
                 weighting="prelec_log_elevation", alpha_dist="lognormal",
                 desperation_rule="rank", kappa=0.5,
                 pareto_shape=3, pareto_scale=200, seed=None,
                 victim_mode=None, sync=None):
        super().__init__(seed=seed)
        self.victim_mode, self.sync = resolve_victim_options(victim_mode, sync)
        ss = (np.random.SeedSequence([int(seed), _VICTIM_STREAM]) if seed is not None
              else np.random.SeedSequence())
        self.victim_rng = np.random.default_rng(ss)  # never touches the global stream
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
        self._agent_list = list(self.agents)
        for i, a in enumerate(self._agent_list):
            a.index = i
        self.victim_history = []
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Proportion crime": crime_proportion,
                "Average Wealth": average_wealth,
                "P90-P10": p90_p10,
            },
            agent_reporters={
                "Wealth": "wealth", "Decision": "decision", "Rank": "income_rank",
                "Desperate": "desperate_state", "Position": "pos", "Caught": "caught",
                "Offended": "offended", "Victimized": "victimized",
                "AmountLost": "amount_lost",
                "OffenderIDs": lambda a: ";".join(str(i) for i in a.offender_ids),
                "VictimID": "victim_id", "AmountStolen": "amount_stolen",
            },
        )
        self.running = True
        self.datacollector.collect(self)

    # --- victimization -------------------------------------------------------
    def draw_victim(self, offender):
        """One victim drawn uniformly, never the offender. Returns None only if the
        local comparison sample contains no other agent (degenerate grids)."""
        if self.victim_mode == "global":
            i = int(self.victim_rng.integers(len(self._agent_list) - 1))
            if i >= offender.index:
                i += 1
            return self._agent_list[i]
        if self.victim_mode == "local":
            cands = [a for a in offender.comparison_sample if a is not offender]
            if not cands:
                return None
            return cands[int(self.victim_rng.integers(len(cands)))]
        raise RuntimeError("draw_victim called with victim_mode 'none'")

    def transfer(self, offender, victim):
        """Move min(R1, max(victim wealth, 0)) from victim to offender and log it."""
        amount = min(float(offender.reward_rb), max(float(victim.wealth), 0.0))
        victim.wealth -= amount
        offender.wealth += amount
        victim.victimized = 1
        victim.amount_lost += amount
        victim.offender_ids.append(offender.unique_id)
        offender.victim_id = victim.unique_id
        offender.amount_stolen = amount
        if victim.done:  # victim already stepped this round (asynchronous schedule)
            victim.wealth_end = victim.wealth
        return amount

    def _step_sync(self):
        order = list(self.agents.shuffle())
        for a in order:      # 1. decisions and all draws, from start-of-step wealth
            a.decide()
        for a in order:      # 2. thefts, capped at the victim's running wealth
            if a.pending_victim is not None:
                self.transfer(a, a.pending_victim)
        for a in order:      # 3. own income or penalty, then living cost
            a.settle()
            a.done = True

    def advance(self):
        """One step without data collection (used by step() and the batch scripts)."""
        for a in self._agent_list:
            a.reset_step_log()
        if self.sync:
            self._step_sync()
        else:
            self.agents.shuffle_do("step")
        n = len(self._agent_list)
        thefts = [a.amount_stolen for a in self._agent_list if a.victim_id != -1]
        self.victim_history.append({
            "victim_rate": sum(a.victimized for a in self._agent_list) / n,
            "thefts": len(thefts),
            "transferred": float(sum(thefts)),
            "zero_transfers": sum(1 for x in thefts if x == 0.0),
            "n_agents": n,
        })

    def victim_summary(self):
        """Run-level victimization statistics for run_summary.summarize."""
        h = self.victim_history
        if not h:
            return None
        thefts = sum(x["thefts"] for x in h)
        total = sum(x["transferred"] for x in h)
        last = h[-1]
        return {
            "victim_rate_final": last["victim_rate"],
            "victim_rate_mean": float(np.mean([x["victim_rate"] for x in h])),
            "mean_loss_final": last["transferred"] / last["n_agents"],
            "mean_transfer": total / thefts if thefts else float("nan"),
            "mean_transfer_final": (last["transferred"] / last["thefts"]
                                    if last["thefts"] else float("nan")),
            "zero_transfer_share": (sum(x["zero_transfers"] for x in h) / thefts
                                    if thefts else float("nan")),
        }

    def step(self):
        self.advance()
        self.datacollector.collect(self)
