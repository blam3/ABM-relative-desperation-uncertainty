# archive/june-2026-model/

Code kept for reproducibility, not for new runs.

The version of the model that produced the results in the June 2026 manuscript
(`archive/manuscripts-2026/SPPS-Manuscript-ABM-rule-breakingabs.docx`): the corrected binary-lottery sign
(June 3), the neighbor-radius fix (May 15), and the reference-shift desperation
mechanism.

These files were copied from the versions stored in the Claude project for this
manuscript on 2026-09-18, not from the cluster. They were checked against the
model used for the September pilot runs and reproduce it agent for agent
(identical wealth, decisions, beta and alpha after 8 steps on a 25 x 25 grid with
a shared seed). If the cluster copies differ, the cluster copies are the record
of what was run.

Three problems in this code are fixed in the current version at the repository
root (see CHANGELOG.md and PARAMETERS.md):

1. `prelec()` uses beta directly as Prelec's elevation parameter, so beta = 0
   gives w(p) = 1 and beta = -1 gives w(p) > 1.
2. alpha is drawn from a clipped t-distribution, which puts about 21% of agents
   at exactly 0.1 in the alpha = 0.5 conditions.
3. `batch_run_hpc.py` and `batch_run_sensitivity.py` reuse random seeds across
   conditions (815 distinct seeds for 4,500 runs).

To reproduce the June results, run these files as they are. To reproduce them
with the current code instead, pass `weighting="prelec_raw"`, `alpha_dist="t_clip"`, `lambd=0.2`,
and `alpha_scale=0.5` (see `archive/pilot-sept-2026/equiv_test.py`).
