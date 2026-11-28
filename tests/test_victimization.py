"""Tests for victim_mode ("none" / "global" / "local") and the synchronous schedule.

1. Regression: victim_mode="none" reproduces tag jasss-v1 exactly (3 seeds x 2 conditions).
2. Wealth accounting in both victim arms, asynchronous and synchronous, for 50 steps.
3. Sampling: no self-victimization, local victims come from the comparison sample,
   transfers are never negative (10,000 draws per arm, plus the logs of a full run).
4. Equilibrium smoke test (local arm, default condition, 50 steps). Skipped unless
   DMURD_SMOKE=1, because it runs a 100 x 100 grid.

Run from the repository root:
    python -m unittest tests.test_victimization -v
"""
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(REPO, "tests", "_trajectory_driver.py")
BASELINE_TAG = "jasss-v1"

from model import rel_DMAP_model, resolve_victim_options  # noqa: E402

OPTS = dict(lambd=0.5, gamma=0.3, reward_rb=50, reward_rf=25, cost_rb=250,
            beta_scale=0.5, alpha_scale=0.35, weighting="prelec_log_elevation",
            alpha_dist="lognormal", desperation_rule="rank", min_start_wealth=100,
            pareto_shape=3, pareto_scale=200, p=0.8, num_neighbors=20,
            tau_softmax=3.0, theta_softmax=1.5)
R1 = OPTS["reward_rb"]


def make_model(victim_mode, sync=False, seed=7, size=20, beta=0, alpha=1, nu=0.35):
    np.random.seed(seed)
    return rel_DMAP_model(beta_loc=beta, alpha_loc=alpha, income_rank_threshold=nu,
                          width=size, height=size, seed=seed, victim_mode=victim_mode,
                          sync=sync, **OPTS)


def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("DMURD_")}
    return env


# --------------------------------------------------------------------------- 1
class RegressionAgainstBaseline(unittest.TestCase):
    """victim_mode="none" (default and explicit) is identical to jasss-v1."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dmurd_regress_")
        base = os.path.join(cls.tmp, "baseline")
        os.makedirs(base)
        try:
            archive = subprocess.run(["git", "-C", REPO, "archive", BASELINE_TAG],
                                     check=True, capture_output=True).stdout
        except subprocess.CalledProcessError as e:
            raise unittest.SkipTest(f"tag {BASELINE_TAG} not available: {e.stderr.decode()}")
        subprocess.run(["tar", "-x", "-C", base], input=archive, check=True)
        cls.results = {}
        for name, tree, flags in [("baseline", base, []),
                                  ("new_default", REPO, []),
                                  ("new_explicit", REPO, ["--explicit-none"])]:
            out = os.path.join(cls.tmp, name + ".pkl")
            subprocess.run([sys.executable, "-I", DRIVER, "--tree", tree, "--out", out] + flags,
                           check=True, env=clean_env(), cwd=cls.tmp)
            with open(out, "rb") as f:
                cls.results[name] = pickle.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _compare(self, name):
        old, new = self.results["baseline"]["runs"], self.results[name]["runs"]
        self.assertEqual(sorted(old), sorted(new))
        self.assertEqual(len(old), 6)  # 2 conditions x 3 seeds
        for key in old:
            o, n = old[key], new[key]
            self.assertEqual(o["seed"], n["seed"])
            for field, arr in o["traj"].items():
                np.testing.assert_array_equal(arr, n["traj"][field], err_msg=f"{key} {field}")
            ncols = len(o["model_var_cols"])
            self.assertEqual(o["model_var_cols"], n["model_var_cols"][:ncols])
            np.testing.assert_array_equal(o["model_vars"], n["model_vars"][:, :ncols])
            # summary row: every jasss-v1 column identical, new columns appended only
            k = len(o["summary_cols"])
            self.assertEqual(o["summary_cols"], n["summary_cols"][:k], key)
            self.assertEqual(len(o["summary"]), k)
            for col, a, b in zip(o["summary_cols"], o["summary"], n["summary"][:k]):
                same = (a == b) or (isinstance(a, float) and math.isnan(a)
                                    and isinstance(b, float) and math.isnan(b))
                self.assertTrue(same, f"{key} summary column {col}: {a!r} vs {b!r}")
            extra = dict(zip(n["summary_cols"][k:], n["summary"][k:]))
            self.assertEqual(extra["Sync"], 0)
            self.assertEqual(extra["Victim_Rate_Final"], 0.0)
            self.assertTrue(math.isnan(extra["Mean_Transfer"]))
            # raw per-agent rows: original columns identical
            h = len(o["raw_header"])
            self.assertEqual(o["raw_header"], n["raw_header"][:h])
            self.assertEqual(len(o["raw"]), len(n["raw"]))
            for ro, rn in zip(o["raw"], n["raw"]):
                self.assertEqual(list(ro), list(rn[:h]), key)

    def test_default_none_matches_baseline(self):
        self._compare("new_default")

    def test_explicit_none_matches_baseline(self):
        self._compare("new_explicit")


# --------------------------------------------------------------------------- 2
class WealthAccounting(unittest.TestCase):
    """Change in total wealth = law-abiding income - living costs - penalties."""

    STEPS = 50

    def _run(self, victim_mode, sync):
        m = make_model(victim_mode, sync, seed=11)
        agents = list(m.agents)
        thefts = 0
        for step in range(self.STEPS):
            before = {a.unique_id: float(a.wealth) for a in agents}
            total_before = math.fsum(before.values())
            m.step()
            total_after = math.fsum(float(a.wealth) for a in agents)
            income = math.fsum(a.income for a in agents)
            living = math.fsum(a.living_cost for a in agents)
            penalty = math.fsum(a.penalty for a in agents)
            unfunded = math.fsum(a.unfunded_gain for a in agents)
            stolen = math.fsum(a.amount_stolen for a in agents)
            lost = math.fsum(a.amount_lost for a in agents)
            msg = f"{victim_mode} sync={sync} step {step + 1}"
            if victim_mode == "none":
                expected = income - living - penalty + unfunded
            else:
                self.assertEqual(unfunded, 0.0, msg)
                expected = income - living - penalty
            self.assertTrue(math.isclose(total_after - total_before, expected, rel_tol=1e-9),
                            f"{msg}: dW={total_after - total_before!r}, expected={expected!r}")
            self.assertTrue(math.isclose(stolen, lost, rel_tol=1e-9, abs_tol=1e-9), msg)
            # per agent: own gain is exactly income - penalty - living + stolen - lost
            for a in agents:
                d = float(a.wealth) - before[a.unique_id]
                e = (a.income - a.penalty - a.living_cost + a.amount_stolen - a.amount_lost
                     + a.unfunded_gain)
                self.assertTrue(math.isclose(d, e, rel_tol=1e-9, abs_tol=1e-9),
                                f"{msg} agent {a.unique_id}: {d!r} vs {e!r}")
                if a.victim_id != -1:
                    self.assertEqual(a.offended, 1)
                    self.assertFalse(a.detected)
            thefts += sum(1 for a in agents if a.victim_id != -1)
        if victim_mode != "none":
            self.assertGreater(thefts, 0, "no thefts occurred, so the test proves nothing")
        return thefts

    def test_global_async(self):
        self._run("global", False)

    def test_local_async(self):
        self._run("local", False)

    def test_global_sync(self):
        self._run("global", True)

    def test_local_sync(self):
        self._run("local", True)

    def test_none_control_leaks_R1(self):
        """Control: in "none" the identity needs the unfunded R1 term."""
        m = make_model("none", False, seed=11)
        agents = list(m.agents)
        before = math.fsum(float(a.wealth) for a in agents)
        m.step()
        after = math.fsum(float(a.wealth) for a in agents)
        unfunded = math.fsum(a.unfunded_gain for a in agents)
        self.assertGreater(unfunded, 0)
        no_leak = (math.fsum(a.income for a in agents) - math.fsum(a.living_cost for a in agents)
                   - math.fsum(a.penalty for a in agents))
        self.assertFalse(math.isclose(after - before, no_leak, rel_tol=1e-9))


# --------------------------------------------------------------------------- 3
class Sampling(unittest.TestCase):
    DRAWS = 10_000

    def _draws(self, victim_mode):
        m = make_model(victim_mode, seed=5)
        agents = list(m.agents)
        rng = np.random.default_rng(123)
        for a in rng.choice(len(agents), size=int(0.3 * len(agents)), replace=False):
            agents[a].wealth = -float(rng.uniform(1, 500))
        n_zero = 0
        for _ in range(self.DRAWS):
            off = agents[int(rng.integers(len(agents)))]
            off.relative_desperation()
            v = m.draw_victim(off)
            self.assertIsNotNone(v)
            self.assertIsNot(v, off)
            self.assertNotEqual(v.unique_id, off.unique_id)
            if victim_mode == "local":
                self.assertTrue(any(v is s for s in off.comparison_sample))
            w_before = float(v.wealth)
            amount = m.transfer(off, v)
            self.assertGreaterEqual(amount, 0.0)
            self.assertLessEqual(amount, R1)
            if w_before <= 0:
                self.assertEqual(amount, 0.0)
                self.assertEqual(float(v.wealth), w_before)
            else:
                self.assertGreaterEqual(float(v.wealth), 0.0)
            n_zero += amount == 0.0
        self.assertGreater(n_zero, 0, "negative-wealth victims were never drawn")

    def test_global_draws(self):
        self._draws("global")

    def test_local_draws(self):
        self._draws("local")

    def test_logged_thefts_over_a_run(self):
        for mode in ("global", "local"):
            for sync in (False, True):
                m = make_model(mode, sync, seed=3)
                agents = list(m.agents)
                by_id = {a.unique_id: a for a in agents}
                for _ in range(50):
                    m.step()
                    for a in agents:
                        self.assertGreaterEqual(a.amount_stolen, 0.0)
                        self.assertGreaterEqual(a.amount_lost, 0.0)
                        self.assertNotIn(a.unique_id, a.offender_ids)
                        if a.victim_id == -1:
                            continue
                        self.assertNotEqual(a.victim_id, a.unique_id)
                        self.assertIn(a.unique_id, by_id[a.victim_id].offender_ids)
                        if mode == "local":
                            self.assertIn(a.victim_id,
                                          {s.unique_id for s in a.comparison_sample})
                    self.assertEqual(sum(len(a.offender_ids) for a in agents),
                                     sum(a.victim_id != -1 for a in agents))


class Options(unittest.TestCase):
    def test_env_and_argument_resolution(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DMURD_VICTIM", None)
            os.environ.pop("DMURD_SYNC", None)
            self.assertEqual(resolve_victim_options(), ("none", False))
            os.environ["DMURD_VICTIM"] = "local"
            os.environ["DMURD_SYNC"] = "1"
            self.assertEqual(resolve_victim_options(), ("local", True))
            self.assertEqual(resolve_victim_options("global", False), ("global", False))
            m = make_model(None, None, size=8)
            self.assertEqual((m.victim_mode, m.sync), ("local", True))
            os.environ["DMURD_VICTIM"] = "everyone"
            with self.assertRaises(ValueError):
                resolve_victim_options()
            os.environ["DMURD_VICTIM"] = "none"
            os.environ["DMURD_SYNC"] = "yes"
            with self.assertRaises(ValueError):
                resolve_victim_options()


# --------------------------------------------------------------------------- 4
@unittest.skipUnless(os.environ.get("DMURD_SMOKE") == "1", "set DMURD_SMOKE=1 to run")
class EquilibriumSmoke(unittest.TestCase):
    def test_local_default_condition_50_steps(self):
        import run_summary as rs
        import itertools
        combos = list(itertools.product([-1, 0, 1], [0.5, 1, 2], [0.05, 0.2, 0.35, 0.5, 0.65]))
        cond = combos.index((0, 1, 0.35)) + 1  # batch_run_hpc.ALL_COMBINATIONS numbering
        seed = int(np.random.SeedSequence([20260915, cond, 0]).generate_state(1)[0])
        rows = {}
        for mode in ("local", "none"):
            m = make_model(mode, False, seed=seed, size=100)
            agents = list(m.agents)
            crime = []
            for _ in range(50):
                m.advance()
                crime.append(rs.step_stats(agents)[0])
            vr = [h["victim_rate"] for h in m.victim_history]
            rows[mode] = (crime[24], crime[49], vr[24], vr[49], m.victim_summary())
            self.assertTrue(all(np.isfinite(crime)))
        print(f"\nSMOKE seed={seed} grid=100x100 beta=0 alpha=1 nu=0.35 lambda=0.5")
        for mode, (c25, c50, v25, v50, vs) in rows.items():
            print(f"SMOKE {mode}: offending step25={c25:.4f} step50={c50:.4f} "
                  f"victim_rate step25={v25:.4f} step50={v50:.4f} "
                  f"mean_transfer={vs['mean_transfer']:.3f} "
                  f"zero_transfer_share={vs['zero_transfer_share']:.4f}")


if __name__ == "__main__":
    unittest.main()
