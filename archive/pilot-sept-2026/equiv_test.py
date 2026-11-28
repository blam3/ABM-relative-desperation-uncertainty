"""Check that the current model in legacy mode reproduces the June 2026 code.

Runs both versions on the same grid with the same seed and compares every agent's
wealth, decision, beta and alpha. Run from this folder: python equiv_test.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURRENT = os.path.join(HERE, "..", "..")
JUNE = os.path.join(HERE, "..", "june-2026-model")


def run(path, kwargs):
    for m in ("model", "agents", "cognitive_functions"):
        sys.modules.pop(m, None)
    sys.path.insert(0, path)
    from model import rel_DMAP_model
    np.random.seed(777)
    m = rel_DMAP_model(width=30, height=30, beta_loc=-1, alpha_loc=0.5,
                       income_rank_threshold=0.35, seed=777, **kwargs)
    for _ in range(10):
        m.step()
    out = sorted((a.unique_id, float(a.wealth), a.decision, float(a.beta), float(a.alpha))
                 for a in m.agents)
    sys.path.pop(0)
    return np.array(out, dtype=float)


if __name__ == "__main__":
    june = run(JUNE, dict(use_refshift=True))
    now = run(CURRENT, dict(weighting="prelec_raw", alpha_dist="t_clip", alpha_scale=0.5,
                           lambd=0.2))
    same = np.allclose(june, now)
    print(f"identical: {same} | max absolute difference {np.abs(june - now).max()}")
    sys.exit(0 if same else 1)
