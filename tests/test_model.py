import unittest

import numpy as np
import pandas as pd

import cognitive_functions as cf
from run import simulate


class ModelChecks(unittest.TestCase):
    def test_weighting_identity_bounds_and_monotonicity(self):
        probabilities = np.linspace(0, 1, 101)
        self.assertTrue(np.allclose([cf.prelec(0, 1, p) for p in probabilities], probabilities))
        for beta in (-10, -1, 0, 1, 10):
            weights = np.array([cf.prelec(beta, 0.7, p) for p in probabilities])
            self.assertTrue(np.all((weights >= 0) & (weights <= 1)))
            self.assertTrue(np.all(np.diff(weights) >= 0))

    def test_seed_reproduces_complete_trajectory(self):
        first = simulate(8, 8, 3, seed=11)
        second = simulate(8, 8, 3, seed=11)
        pd.testing.assert_frame_equal(first.datacollector.get_model_vars_dataframe(), second.datacollector.get_model_vars_dataframe())
        self.assertEqual([(a.wealth, a.decision) for a in first.agents], [(a.wealth, a.decision) for a in second.agents])
        other = simulate(8, 8, 3, seed=12)
        self.assertFalse(first.datacollector.get_model_vars_dataframe().equals(other.datacollector.get_model_vars_dataframe()))

    def test_model_output_invariants(self):
        model = simulate(8, 8, 3, seed=11)
        frame = model.datacollector.get_model_vars_dataframe()
        self.assertEqual(len(model.agents), 64)
        self.assertEqual(len(frame), 4)
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertTrue(frame["Proportion crime"].between(0, 1).all())
        self.assertTrue((frame["P90-P10"] >= 0).all())

    def test_invalid_demo_sizes_fail(self):
        for args in ((1, 8, 2), (8, 1, 2), (8, 8, 0)):
            with self.assertRaises(ValueError):
                simulate(*args)


if __name__ == "__main__":
    unittest.main()
