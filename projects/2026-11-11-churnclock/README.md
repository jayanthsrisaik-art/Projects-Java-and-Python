# churnclock – churn time-to-event modelling without censoring bias or leakage

## Problem
A subscription business wants to know, for each customer at **day 30 after signup**, how likely
they are to cancel in the next 90 / 180 / 365 days and how long they are likely to stay.
The data makes this harder than it looks:

* **Right censoring.** Most customers have not cancelled *yet*. Someone who signed up 60 days ago
  and is still active has not "retained for a year". Recent signups have short follow-up because
  signups are staggered, and some accounts close for non-churn reasons (for example, a merge into an
  enterprise contract), which also ends observation.
* **Two common shortcuts are biased.** Using "cancelled within H days?" as a classification label
  and treating censored customers as retained under-counts churn. Dropping customers whose status at H
  is unknown keeps the short-lived ones and drops the long-lived ones, so it over-counts churn.
* **Leakage.** Lifetime aggregates such as total logins or tickets look predictive. They grow with
  the time a customer stayed, so they encode the outcome and are not available at day 30.

churnclock frames the task as **landmark survival analysis**. The cohort is every customer still
active and observed at day 30. Features come only from signup attributes and the first 30 days.
The target is time from the landmark to cancellation, right-censored. It compares a Cox
proportional-hazards model, written from scratch in NumPy, against a Kaplan–Meier baseline, both
logistic shortcuts, a deliberately leaky model, and an oracle that knows the true hazard. All
models are scored with censoring-aware metrics.

## Architecture
```
churnclock/synth.py     synthetic subscription business: customers.csv, usage.csv (one row per
                        30-day period since signup), truth.csv (generator internals, oracle only).
                        Post-landmark churn is Weibull PH with known log-hazard ratios; staggered
                        signups -> administrative censoring; exponential non-churn closures.
churnclock/features.py  durations, landmark cohort (observed > 30 days), day-30 features
                        (period-0 usage only), and the LEAKY lifetime aggregates for the ablation
churnclock/survival.py  Kaplan-Meier (S(t) and S(t-)); Cox PH: Breslow ties, L2, damped Newton,
                        Breslow baseline cumulative hazard, survival curves, median, JSON save/load
churnclock/metrics.py   Harrell's C, Uno's IPCW C (truncated at 365 d), IPCW time-dependent Brier score (Graf), integrated Brier score,
                        calibration table (mean predicted vs per-bin Kaplan-Meier observed churn)
churnclock/pipeline.py  hash-based customer split, model zoo, evaluation, report
churnclock/cli.py       generate | run | seeds | predict
```
Flow: `customers + usage → landmark cohort + day-30 features → customer-level split →
fit on train (scalers, logistic models, Cox and the censoring distribution G all from train only)
→ predict survival on test → C-index / Brier@h / IBS / calibration`.

## Setup
```
pip install -r requirements.txt     # tested: Python 3.11, numpy 2.4, pandas 3.1, scikit-learn 1.9, scipy 1.17
python -m pytest -q                 # 30 tests, ~5 s
```

## Usage
```
python -m churnclock generate --out data            # 6,000 customers, seed 7
python -m churnclock run --data data --out reports  # metrics.json, metrics.txt, cox.model.json
python -m churnclock seeds --seeds 1,2,3,4,5        # repeat on 5 independently generated datasets
python -m churnclock predict --model samples/cox.model.json \
    --customers samples/new_customers.csv --usage samples/new_usage.csv --out preds.csv
```
`predict` reads only period-0 usage rows; later rows are ignored, and a test checks this. Its output lists
`risk_score` (log relative hazard), `churn_by_{90,180,365}d`, and `median_days_after_landmark`
(`never_below_0.5` when the predicted curve stays above 0.5 within observed follow-up). It also includes
`has_period0_usage`, which flags customers with no usage rows; those are scored as zero activity.
Missing input columns or non-positive horizons stop `predict` with exit code 2 and a message naming the problem.

### Samples
| file | what |
|---|---|
| `samples/new_customers.csv`, `samples/new_usage.csv` | six hand-written day-30 customers (N005 has a period-1 row that must be ignored; N006 has no usage) |
| `samples/cox.model.json` | Cox model produced by `run` on the default data |
| `samples/expected_predictions.csv` | `predict` output for the six customers |
| `samples/expected_metrics.json` / `.txt` | `run` output for `generate` defaults (seed 7) |
| `samples/expected_seeds.txt` | `seeds` output for data seeds 1–5 |

## Measured results
Default data (seed 7): 6,000 customers. 808 were excluded because they cancelled or stopped being
observed before day 30. The landmark cohort is 5,192, split into 3,906 train and 1,286 test.
Test event rate: 42%. Brier scores (BS) are IPCW at days after the landmark, and IBS is integrated
over days 15–540. `pred@180` is the mean predicted churn by day 180, and `KM@180` is the Kaplan–Meier
churn of the test set. `cal@180` is the size-weighted absolute calibration error over 5 risk quintiles.

| model | C-index | BS@90 | BS@180 | BS@365 | IBS | pred@180 | KM@180 | cal@180 |
|---|---|---|---|---|---|---|---|---|
| Kaplan–Meier (no covariates) | 0.5000 | 0.1356 | 0.2105 | 0.2482 | 0.2078 | 0.2986 | 0.3065 | – |
| logistic, censored = retained | 0.6921 | 0.1260 | 0.1811 | 0.2202 | 0.1880 | 0.2534 | 0.3065 | 0.0536 |
| logistic, drop censored | 0.6928 | 0.1251 | 0.1779 | 0.1936 | 0.1768 | 0.3437 | 0.3065 | 0.0368 |
| **Cox PH (day-30 features)** | 0.6945 | 0.1252 | 0.1767 | 0.1870 | **0.1665** | 0.3023 | 0.3065 | **0.0164** |
| Cox PH + lifetime aggregates (LEAKY) | 0.9792 | 0.0018 | 0.0026 | 0.0065 | 0.0070 | 0.5237 | 0.3065 | 0.0258 |
| oracle (true hazard, uses latent engagement) | 0.7069 | 0.1223 | 0.1729 | 0.1790 | 0.1615 | 0.3098 | 0.3065 | 0.0282 |

Across 5 independently generated datasets (`samples/expected_seeds.txt`):
* **Ranking is not where survival modelling wins.** The C-index of Cox and both logistic models is within 0.002 of each
  other on every seed (0.0024 on the default seed-7 data). All three sit 0.011–0.023 below the oracle.
* **Cox has the lowest IBS of the non-oracle models on 5/5 seeds**, for example 0.1665 vs 0.1768 for the best
  logistic model on seed 7. Its BS@180 is lowest on 4/5. On seed 2, logistic "censored = retained" is
  slightly better (0.1750 vs 0.1763).
* **The two shortcuts are biased in opposite directions on every seed.** "Censored = retained" predicts
  0.248–0.258 mean churn by day 180, and "drop censored" predicts 0.334–0.350. The KM estimate on the test set is
  0.274–0.320. Cox's mean prediction is within 0.04 of KM on every seed and within 0.01 on 3 of 5 seeds.
  Calibration error at 180 days is lowest for Cox on 4/5 seeds.
* **Leakage is dramatic and invisible to the metrics.** Adding lifetime totals lifts the offline
  C-index to 0.978–0.980 and drives Brier scores near zero, yet the features do not exist at prediction time.
  The leaky model's mean predicted churn (0.51–0.52 vs ~0.30) is the only sign in the report.
* On seed 7, the estimated log-hazard ratios for the directly comparable features land close to the
  generator's: tickets +0.238 (true +0.22), failed payment +0.786 (+0.85), discount +0.452 (+0.45),
  paid social +0.445 (+0.40), referral −0.208 (−0.30).

## Design decisions
* **Landmarking at day 30** keeps features and the start of the clock aligned. Using day-30 features
  from time zero would condition on surviving to day 30 (immortal-time bias). Customers who cancel before
  day 30 are a separate problem: the cohort excludes them, and the report counts them.
* **Customer-level hash split** (`sha256(seed:customer_id)`) is reproducible, independent of row order,
  and keeps each customer on one side. Every fitted quantity comes from train only: scalers, logistic
  models, Cox and its baseline hazard, and the censoring distribution `G` used for IPCW weights.
* **Cox from scratch.** It uses Breslow ties because cancellations are recorded in whole days. Newton steps
  are halved until they increase the log-likelihood, and a small L2 penalty (1.0 on standardised features)
  keeps the solution unique. Tests check it against a brute-force partial likelihood with ties, a
  numerical gradient at the optimum, coefficient recovery on 20k synthetic rows, and Nelson–Aalen when β = 0.
  `lifelines` was the first choice, but it failed to build in the test environment, and a 100-line NumPy
  version removes that dependency.
* **Metrics that handle censoring.** Harrell's C covers ranking. The IPCW Brier score covers both
  calibration and discrimination at fixed horizons, and IBS summarises it. A test checks that IPCW Brier
  matches the plain Brier score when nothing is censored, and recovers the true score within 0.01 under
  heavy random censoring.
* **The logistic models get every advantage:** one model per horizon on the same features. Their
  survival curves are made monotone by a running minimum. The test is whether the *label definition*
  matters, not whether a weak classifier loses.
* **The oracle** uses the generator's true hazard, including latent engagement that no model sees.
  It is a ceiling, not a competitor.

## Limitations
* The data is synthetic, and the generator satisfies proportional hazards by construction, so Cox is
  correctly specified. With real churn, effects often change over tenure (for example, a discount ending).
  The next step would be stratified or time-varying Cox, or a non-PH model (random survival forest or
  gradient-boosted survival), checked with Schoenfeld residuals.
* Censoring is assumed independent of churn given the features, which IPCW and Cox both require. A
  closure that is really churn under another name would break this.
* `monthly_price` is almost a function of plan and discount. The plan coefficients must therefore be read
  together with the price coefficient (team: −0.123 − 0.008 × 67 ≈ −0.66 vs true −0.60). The model's
  predictions are unaffected.
* There is one landmark only. Re-scoring customers at day 60, 90 and later would need stacked landmark
  datasets.
* IBS stops at day 540 because past that point few test customers remain at risk and the IPCW weights
  become unstable. Harrell's C depends on the censoring pattern, so `metrics.json` also reports Uno's IPCW C truncated at day 365 (`uno_c_365`; Cox 0.6974 vs Harrell 0.6945 on seed 7). It assumes the censoring distribution from train applies to test.
* Breslow's approximation is slightly biased when ties are heavy. Efron's method would be more accurate.
