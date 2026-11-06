# Parameter choices and what they mean

Every value in the simulation study, why it takes that value, and what the three
levels of uncertainty aversion now correspond to. Symbols follow the manuscript.
Values are set in `batch_run_hpc.py` (`MODEL_OPTIONS`) unless noted.

Last updated 2026-09-29, after the desperation-severity (lambda) sweep and
recalibration described in `CHANGELOG.md`. Wherever this file and the June 2026
manuscript disagree, this file describes the code in this repository and the
manuscript still describes the June sweep.

## 1. Summary table

| Symbol | Parameter | Value(s) | Why this value |
|---|---|---|---|
| N | Population | 10,000 | One agent per cell of the grid; large enough that a condition mean is stable across replications. |
| - | Grid | 100 x 100 torus | No edges, so no agent has a smaller comparison pool than any other. |
| n_nbr | Comparison group | 20 sampled from the 48 cells within a Moore radius of 3 | Local rather than global comparison; 20 approximates the size of the network within which income comparisons predict deprivation and health ([Mangyo & Park, 2011](https://muse.jhu.edu/article/466987)). |
| - | Steps | 25 (50 in the convergence check) | Offending stabilizes well before step 25; the supplement reports 50-step runs. |
| k, s, theta_min | Initial wealth: Lomax shape, scale, minimum | 3, 200, 100 | Right-skewed wealth as in modern economies ([Jenkins, 2017](https://doi.org/10.1111/ecca.12217)); gives an initial Gini of about .30 and an equilibrium Gini near .40, in the range of US estimates ([Boal, 2026](https://doi.org/10.1177/05694345251393682)). The minimum of 100 lets every agent cover several steps of living costs. |
| R0 | Law-abiding reward | 25 | Legal income per step. |
| R1 | Offending reward | 50 | A successful theft pays twice the legal wage. |
| C | Penalty if apprehended | 250 | Ten times the legal wage, so apprehension is a serious setback. |
| p | Probability of not being apprehended | 0.80 | About 17% of property crimes in the United States were cleared by arrest or exceptional means in 2019 ([FBI, 2020](https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/topic-pages/clearances)). |
| rho | Living cost rate | Normal(0.80, 0.10), truncated to [0.10, 1.00], times R0 | Costs scale with legal income; agents who never offend gain about 5 units per step. |
| gamma | Utility curvature | 0.3 | Strong diminishing sensitivity to gains and losses, the property that makes outcomes far from the reference point feel alike. The value function is sign(x)|x|^gamma so that debt is defined. |
| tau | Satisficing threshold | 3.0 | Agents default to law-abiding behavior and need a clear subjective advantage before offending (Luce, 1959; [Musslick & Masis, 2023](https://doi.org/10.1111/cogs.13259)). Swept over {1, 3, 5} in the sensitivity analysis. |
| eta | Decision noise | 1.5 | Choice is probabilistic rather than deterministic. Swept over {0.5, 1.5, 3.0}. |
| nu | Relative desperation threshold | .05, .20, .35, .50, .65 | Income-rank percentiles from the poorest in the neighborhood to the lower-middle of it. |
| lambda | Desperation severity | 0.5 | Weight on the felt wealth gap in a desperate agent's reference point. Chosen from a 5-level sweep (0.05, 0.1, 0.2, 0.5, 1.0) at beta = 0, alpha = 1: values >= 0.5 make the per-capita offending rate among desperate agents rise monotonically with the desperation threshold nu, as the pre-registered hypothesis predicts. See Section 6. |
| beta_mu | Mean uncertainty aversion | -1, 0, 1 | Optimistic, undistorted, and pessimistic evaluation of the odds (Section 3). |
| beta_sigma | SD of beta | 0.5 (normal) | Agents within a condition differ; 95% of them fall between -0.98 and 0.98 of the condition mean. |
| alpha_mu | Median likelihood sensitivity | 0.5, 1, 2 | Insensitive, undistorted, and oversensitive probability perception (Section 4). |
| alpha_sigma | Log-scale SD of alpha | 0.35 (log-normal) | Keeps alpha positive without clipping; 95% of agents fall within a factor of 1.99 of the condition median. |
| - | Replications | 100 per condition, 45 conditions | 4,500 runs; each run has its own random stream (Section 6). |

## 2. The decision an agent faces

Each step an agent compares the subjective value of offending,

    SV_off = w(p) u(theta + R1) + (1 - w(p)) u(theta - C),

with the subjective value of law-abiding behavior, SV_law = u(theta + R0), and
offends with probability logistic((SV_off - SV_law - tau) / eta). A desperate
agent evaluates both options relative to a reference point R = theta + lambda D,
where D is its mean upward wealth gap to richer neighbors ([Yitzhaki, 1979](https://doi.org/10.2307/1883197)).

With the payoffs above, offending loses money on average: 0.80 x 50 - 0.20 x 250
= -10 units per step, against +5 units per step for law-abiding behavior after
living costs. This is deliberate. Property crime is often not the payoff-maximizing
choice ([Tittle et al., 2010](https://doi.org/10.1016/j.ssresearch.2010.08.007)), so
any offending the model produces comes from how agents evaluate the odds and from
how far behind their neighbors they are, not from the arithmetic of the payoffs.

## 3. Uncertainty aversion (beta): what the three levels mean now

The model uses Prelec's two-parameter weighting function
([Prelec, 1998](https://doi.org/10.2307/2998573)):

    w(p) = exp(-delta (-ln p)^alpha),   delta = exp(beta).

Entering beta as an exponent is the September 2026 fix. Prelec's elevation
parameter delta must be positive, and delta = 1 is the point at which the function
introduces no optimism or pessimism. Writing delta = exp(beta) puts that neutral
point at beta = 0, keeps the symmetric design (-1, 0, 1), and keeps every decision
weight inside (0, 1) for any real beta.

Decision weights for a successful theft (p = .80) under the current code:

| beta | delta = exp(beta) | w(.80), alpha = 0.5 | alpha = 1 | alpha = 2 | Weight on apprehension, 1 - w(.80), at alpha = 1 |
|---|---|---|---|---|---|
| -1 | 0.37 | 0.84 | 0.92 | 0.98 | 0.08 |
| 0 | 1.00 | 0.62 | 0.80 | 0.95 | 0.20 |
| +1 | 2.72 | 0.28 | 0.55 | 0.87 | 0.45 |

Read the three levels as follows, at alpha = 1, where the objective chance of
apprehension is .20:

- **beta = -1 (uncertainty seeking, optimistic).** The agent behaves as though a
  successful theft were more likely than it is (w = 0.92) and as though
  apprehension carried only 0.08 of the weight, half its objective value.
- **beta = 0 (undistorted).** Decision weights equal the objective probabilities:
  w(.80) = .80 and apprehension carries its objective weight of .20. This is the
  baseline the manuscript describes as neutral, and with this parameterization it
  finally is one.
- **beta = +1 (uncertainty averse, pessimistic).** The agent behaves as though a
  successful theft were unlikely (w = 0.55) and treats apprehension as
  more than twice as likely as it is (0.45 against .20).

For comparison, the same values under the June 2026 code, where beta entered
directly as delta:

| beta | delta = beta | w(.80), alpha = 0.5 | alpha = 1 | alpha = 2 | Weight on apprehension at alpha = 1 |
|---|---|---|---|---|---|
| -1 | -1.00 | 1.60 | 1.25 | 1.05 | -0.25 |
| 0 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| +1 | 1.00 | 0.62 | 0.80 | 0.95 | 0.20 |

Only beta = 1 with alpha = 1 was undistorted there. At beta = 0 every probability
was weighted 1, so agents treated a successful theft as certain; at beta = -1 the
weights exceeded 1, so the weight on apprehension was negative and being caught
raised the value of offending. About 98% of agents in the beta = -1 conditions and
50% of those in the beta = 0 conditions had weights above 1.

## 4. Likelihood sensitivity (alpha)

alpha is the curvature of the same function. It governs how well an agent tells
probabilities apart, independently of optimism or pessimism
([Wakker, 2010](https://doi.org/10.1017/CBO9780511779329); [Henkel, 2024](https://doi.org/10.1016/j.geb.2024.03.015)):

- **alpha < 1 (likelihood insensitivity).** The inverse-S shape: unlikely outcomes
  are overweighted and likely ones underweighted, so probabilities are pulled
  toward the middle. At alpha = 0.5 and beta = 0, the .20 chance of apprehension
  is weighted 0.38.
- **alpha = 1.** No curvature; with beta = 0 the weights are the probabilities.
- **alpha > 1 (oversensitivity).** Probabilities are pushed away from the middle,
  so a likely success is treated as close to certain (0.95 at beta = 0).

Because higher alpha means *more* sensitivity, results are reported as an
insensitivity index, iota = -log2(alpha), so that alpha = 2, 1, 0.5 map to
iota = -1, 0, +1 and larger values mean more insensitivity.

One consequence is worth stating in advance, because it drives the direction of
the alpha effect in the corrected model: apprehension in this model is unlikely
(p = .80 of success). Insensitivity inflates unlikely outcomes, so it inflates the
chance of being caught, and therefore deters offending. It would encourage
offending only where success itself is unlikely.

## 5. Heterogeneity within a condition

- beta is drawn per agent from a normal distribution centered on the condition mean
  with SD 0.5.
- alpha is drawn from a log-normal distribution with the condition value as its
  median and a log-scale SD of 0.35: 95% of agents fall between
  0.25 and 0.99 when alpha_mu = 0.5,
  0.50 and 1.99 when alpha_mu = 1, and
  1.01 and 3.97 when alpha_mu = 2.

The June code drew alpha from a t-distribution with SD 0.5 and clipped it to
[0.1, 10], which set about 21% of agents in the alpha = 0.5 conditions to exactly
0.1. The log-normal draw keeps alpha positive without a clipping bound and gives
proportional spread at every level. The spread of both parameters is in the range
of the individual differences reported in elicitation studies
([von Gaudecker et al., 2022](https://www.jstor.org/stable/resrep65503)).

## 6. Relative desperation (nu and lambda)

An agent is desperate when its income rank among its 20 sampled neighbors is at or
below nu. Rank is used rather than a magnitude-weighted index because rank predicts
health and social outcomes better ([Gero et al., 2022](https://doi.org/10.1016/j.socscimed.2022.114722)).

Because rank is discrete, the threshold fixes the share of desperate agents:

| nu | Expected share desperate | Reading |
|---|---|---|
| .05 | 0.10 | Poorest in the neighborhood |
| .20 | 0.24 | Bottom quintile |
| .35 | 0.38 | Lower-middle of the neighborhood |
| .50 | 0.52 | Below the local median |
| .65 | 0.67 | Most of the neighborhood |

This is a property of the design, not a result: inequality cannot make more agents
desperate under this rule, only make the desperate ones further behind. That is why
`batch_run_inequality.py` also offers a median-based rule, under which the number of
desperate agents can move with inequality.

lambda sets how much of the felt gap enters the reference point. The value
matters more than it looks: a desperate agent is most tempted when law-abiding
income (25) still leaves it below its reference point but a successful theft (50)
would lift it above, that is, when lambda D is between 25 and 50. Larger gaps push
both options into the loss domain, where gamma = 0.3 flattens the value function and
the two options feel alike. lambda therefore sets where offending peaks as a
function of deprivation.

An explicit lambda x nu sweep was run at the no-distortion centre (beta = 0,
alpha = 1) using `batch_run_lambda_sweep.py`. Because the rank-based rule enlarges
the desperate pool with nu but also dilutes it with agents whose upward gap D is
small, low lambda produces a strongly non-monotonic pattern:

| lambda | Desperate-agent offending across nu = .05 -> .65 | Monotone in nu? |
|---|---|---|
| 0.05 | .191 -> .123 -> .102 -> .091 -> .086 | strictly decreasing |
| 0.10 | .253 -> .203 -> .158 -> .133 -> .120 | strictly decreasing |
| 0.20 | .156 -> .193 -> .182 -> .159 -> .141 | inverse-U (peak at nu = .20) |
| 0.50 | .114 -> .117 -> .122 -> .128 -> .133 | **monotone increasing** |
| 1.00 | .114 -> .114 -> .114 -> .122 -> .131 | monotone increasing |

lambda = 0.5 is the smallest value at which the desperate-agent rate rises with nu
as the theory predicts. It also aligns aggregate crime with nu monotonically (from
.088 at nu = .05 to .123 at nu = .65) and produces a desperation gap that is roughly
constant across nu (~.02-.03) rather than diluting. Higher lambda pushes outcomes so
far into the loss domain that the value function flattens and the desperation gap
shrinks further.

## 7. Seeds

Each run draws its seed from `numpy.random.SeedSequence([ROOT_SEED, condition, rep])`
and writes it to the output. The June scripts used
`12345 + rep + int(|beta| * 100 + alpha * 10 + nu * 1000)`, which produced 815
distinct seeds for 4,500 runs and gave the beta = -1 and beta = +1 conditions
identical random streams.

## 8. What is still not calibrated

- p is a single fixed value, so uncertainty aversion and likelihood sensitivity
  cannot be separated by the data the model generates; varying p is the natural
  next sweep.
- lambda is fixed at 0.5, selected from a five-level sweep (Section 6). The full
  main sweep has not been repeated at intermediate values (0.3, 0.4), which could
  refine the balance between preserving desperation gap magnitude and monotonicity
  in nu.
- alpha = 2 describes an uncommon profile; most empirical estimates fall below 1.
  It is kept so that the design is symmetric on the log scale.
- No parameter is estimated from offending data. Simulation-based inference
  ([Cranmer et al., 2020](https://doi.org/10.1073/pnas.1912789117)) is the planned
  route to calibration.
