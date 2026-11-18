# labellag: training a fraud model when chargebacks arrive weeks late

A fraud model is retrained on recent card transactions, and its labels come from chargebacks.
Chargebacks are slow. In this project's simulator the median report delay is 16 days, or 35
days for small amounts, and the hard cap is 120 days. On the day you retrain, **only about 31%
of the frauds from the last 30 days have been reported yet** (measured: 0.309 averaged over 20
backtest cutoffs). The newest fraud patterns are also the ones you most need the model to learn.

Every option for those young rows has a cost:

| strategy | what it does with rows whose label may not have arrived yet |
|---|---|
| `naive` | treat "no chargeback yet" as legitimate (the default in most pipelines) |
| `mature_only` | drop every row younger than the age by which 90% of frauds are reported |
| `drop_young_negatives` | keep young reported frauds, drop young unlabelled rows |
| `ipw` | keep everything and weight each reported fraud by `1 / F(age)`, where `F` is the estimated share of frauds reported by that age |
| `soft_label` | EM: score each unlabelled row by `P(fraud and not yet reported)` = `p(1-F) / (p(1-F) + 1-p)` and train on soft targets, with a cross-fitted IPW seed |
| `rules` | hand-written risk score, no learning (baseline) |
| `oracle` | same window trained on ground truth. **Not deployable**; it shows how much the delay costs |

`F` is estimated at each cutoff from what was knowable then: the delays of a cohort old
enough (≥120 days) that all its chargebacks are in. That avoids the right-truncation bias you
get from measuring delays on recent frauds. The estimate is stratified by amount quantile
because small frauds are reported later.

## Measured results

`python -m labellag backtest`: 5 simulator seeds × 4 rolling cutoffs (days 240/270/300/330),
180-day training window, 21-day test window right after each cutoff and scored against true
fraud, review budget 0.5% of transactions. Gradient boosting (scikit-learn
`HistGradientBoostingClassifier`) with identical hyperparameters for all strategies.
These are the numbers from `results/backtest_summary.md`:

| method | avg precision | recall @0.5% | precision @0.5% | pred rate / true rate | AP − naive (±SE) | beats naive |
|---|---|---|---|---|---|---|
| rules | 0.0625 | 0.077 | 0.133 | – | −0.085 ± 0.023 | 5/20 |
| naive | 0.1474 | 0.112 | 0.245 | 1.07 | 0 | – |
| mature_only | 0.0734 | 0.064 | 0.136 | 1.25 | −0.074 ± 0.014 | 0/20 |
| drop_young_negatives | 0.1396 | 0.104 | 0.230 | 2.31 | −0.008 ± 0.002 | 2/20 |
| **ipw** | **0.1570** | **0.120** | **0.259** | 1.23 | **+0.0095 ± 0.0023** | **18/20** |
| soft_label | 0.1450 | 0.110 | 0.242 | 1.31 | −0.0025 ± 0.0016 | 6/20 |
| oracle (truth) | 0.1755 | 0.130 | 0.280 | 1.29 | +0.028 ± 0.005 | 20/20 |

Delay-curve check: estimated F(14 days) is 0.310 and the realised value for frauds in the 60
days before each cutoff is 0.306.

What this shows, on this simulator only:

* **Waiting for mature labels is the worst option** (AP 0.073 vs 0.147 naive, 0/20 wins). By the
  time the labels are 90% complete, the active fraud campaigns are months old and the newest
  ones are missing entirely.
* **Naive is a strong baseline for ranking.** A young campaign with 30% of its labels already in
  still stands out against the background rate.
* **IPW gives a small but consistent gain**: +0.0095 AP, better than naive in 18 of 20 paired runs.
  That closes about a third of the naive-to-oracle gap (0.028).
* **The soft-label EM did not beat naive** (6/20). The more elaborate method lost to the simple
  one here. A likely reason is that its targets depend on the seed model's `p`, which
  underestimates new campaigns, so it adds noise without adding signal. It is kept in the repo as
  a measured negative result.
* `drop_young_negatives` inflates the predicted fraud rate 2.3× and ranks slightly worse.
* **Calibration does not separate the methods.** Even the oracle predicts 1.29× the realised test
  rate, because campaigns drift between the training window and the next 21 days. Read the
  "pred rate" column as a drift diagnostic, not a strategy score.

## Architecture

```
simulate.py   synthetic stream: stationary risk + time-boxed fraud campaigns (mcc × channel ×
              amount band), log-normal report delays that are slower for small amounts, capped at 120 days
labels.py     observe(df, cutoff): THE leakage gate. Drops future rows, hides unreported
              chargebacks, removes truth columns. DelayCurve estimation + Bayes posterior
model.py      the five training strategies, the IPW and soft-label datasets, the rule baseline
evaluate.py   metrics (AP, recall/precision at review budget, Brier, rate ratio), rolling
              backtest, paired comparison against naive
cli.py        simulate | maturity | train | score | backtest
```

Training code only ever receives `observe(...)` output. Ground truth (`is_fraud`, `true_p`) is
read in exactly two places: scoring the test window and fitting the oracle.

## Setup

```bash
cd projects/2026-11-18-labellag
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 17 tests, about 20 s
```

## Usage

The input format is two CSVs, the same shape as a real export: `transactions.csv`
(`txn_id, day, amount, mcc, channel, hour, card_age_days, dist_km, velocity_24h, foreign,
new_device`) and `chargebacks.csv` (`txn_id, report_day`) containing only the chargebacks
received so far. `day` is a float day number.

```bash
# how complete are my labels on day 250?
python -m labellag maturity --transactions samples/transactions.csv \
    --chargebacks samples/chargebacks.csv --cutoff 250
# train with delay correction, then score new traffic and flag the top 1% for review
python -m labellag train --transactions samples/transactions.csv \
    --chargebacks samples/chargebacks.csv --cutoff 250 --strategy ipw --out model.pkl
python -m labellag score --model model.pkl --input samples/new_transactions.csv --budget 0.01 --out scores.csv
# full experiment (about 4 minutes on 4 cores)
python -m labellag backtest --out results
# make a larger synthetic dataset
python -m labellag simulate --out data --seed 3
```

### Samples

`samples/` comes from `samples/make_samples.sh` (seed 11, 280 days × 120 transactions per day,
history cut at day 250). The tests rerun the CLI and compare the output with the files below.

* `transactions.csv`, `chargebacks.csv`: history as known on day 250 (33.6k rows, 1.2 MB)
* `new_transactions.csv`: days 250–260, unlabelled
* `expected_maturity.json`: for this small sample the curve falls back to one stratum because
  there are too few matured frauds to split. F(14d) = 0.25, and the 9 frauds reported in the
  last 30 days IPW-complete to about 16.9
* `expected_train.json`, `expected_scores.csv`: IPW model summary and scores with a review flag

## Design decisions

* **A rolling-origin backtest, not a random split.** A random split would put the future in the
  training data and would make label delay invisible. Each cutoff trains on the 180 days before
  it and is tested on the 21 days after it, with true labels.
* **The delay curve comes from a fully matured cohort.** Measuring delays on recently reported
  frauds would over-represent fast reports. Rows older than the 120-day cap have all their labels,
  so their delays form an unbiased sample.
* **IPW clipping at F ≥ 0.1** (maximum weight 10) bounds the variance from frauds reported within
  hours.
* **Cross-fitting in soft-label EM**, so that a row's seed score never comes from a model that was
  trained on that row.
* **Paired comparisons.** Every method scores the same test rows, so "beats naive in k/20" and
  the paired standard error are more informative than comparing means.
* The oracle and the delay-curve check are included so the size of the problem is measured,
  not asserted.

## Limitations

* All data is synthetic, and the conclusions (including IPW > naive) are only shown for this
  generator. Real chargeback processes have friendly fraud, disputes that are never filed,
  reversals and reason codes, and none of those are modelled. Every fraud here is eventually
  reported.
* The delay model assumes `F` depends only on age and amount stratum. If delays drift over time,
  or depend on other features, `F` is misspecified.
* Gains are small in absolute terms (+0.0095 AP), and the standard errors come from only 20
  correlated runs (4 cutoffs share each seed's world).
* There is no hyperparameter search. All strategies share one fixed configuration, which may
  favour some of them.
* `score` loads a pickle. Only load model files you produced yourself.
