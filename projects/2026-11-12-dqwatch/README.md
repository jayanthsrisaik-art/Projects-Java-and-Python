# dqwatch: monitor daily warehouse partitions without drowning in alerts

Broken loads in a warehouse usually don't fail. A partial extract lands 40% of the rows, a
retried job appends the same batch twice, a deploy starts writing `NULL` customer ids, an
upstream team switches `amount` from euros to cents, a mapping bug relabels every PayPal order
as `card`, or the load runs 12 hours late. The table still "succeeds", and the dashboards
quietly go wrong.

The usual protections are hand-written thresholds (`row_count` within ±30% of last week,
`null_rate < 1%`) or a trailing z-score. Thresholds ignore that small tables are noisy. A
z-score ignores weekends and alerts every Saturday. Both page people until they stop reading
the pages.

dqwatch profiles each daily partition with one SQL pass and judges every metric against a
**seasonal, robust, sampling-aware** expectation:

* **Seasonal decomposition.** The expected value is a weekday offset (measured against
  centred weekly means, so it does not depend on the level) plus a deseasonalized recent
  level. Weekends are not anomalies, and trends or real level shifts are absorbed within days.
* **Robust scale.** The band width comes from the MAD of recent residuals, so one bad day
  doesn't widen the band.
* **Sampling-aware noise floors.** Today's value is only an anomaly if it is beyond *today's*
  sampling noise:
  * binomial `sqrt(p(1-p)/n)` for null and duplicate rates
  * standard error `sd/(mean*sqrt(n))` for means
  * a χ²(k−1) critical value at today's n for category-mix PSI

  A 7.5% null rate on 40 rows is noise. On 800 rows it is an incident.
* **Minimum actionable effect.** Each metric family has a `min_delta`, so a large table never
  alerts on a 0.1-point change just because the change is statistically significant.
* **No look-ahead.** Every run replays history in date order, and day *i* only reads days < *i*.
  A test checks that results for a date are identical with or without later partitions.

```
$ python -m dqwatch check --db wh.db --config samples/config.json --since 2025-02-17
checked 2025-02-17..2025-05-05: 20 alert(s)
  2025-02-17 refunds    mean:amount              alert       value=2555 expected=25.08 band=[15.08, 41.29] above expected 25.08 (score +37.9)
  2025-02-19 orders     null_rate:customer_id    alert       value=0.01374 expected=0.003711 band=[-, 0.01371] above expected 0.003711 (score +6.2)
  ...
  2025-03-10 orders     row_count                level_shift value=2230 expected=1656 band=[1442, 1902] level shift accepted after 4 consecutive days; ...
  2025-03-16 refunds    row_count                missing     value=0 expected=65 band=[-, -] partition is empty or absent
  2025-03-18 orders     dup_rate                 alert       value=0.477 expected=0 band=[-, 0.002] above expected 0 (score +1843.2)
```

The full output is in [`samples/demo_check_seed7.txt`](samples/demo_check_seed7.txt), and the
injected incidents it should find are in
[`samples/demo_truth_seed7.json`](samples/demo_truth_seed7.json). All 11 incidents are caught.
Four of the 20 alerts are on the benign +35% volume launch on 2025-03-07: three alerts, then a
`level_shift` notice, then the new level is accepted.

## Architecture

```
 warehouse tables (SQLite)                    dq_metrics / dq_category_counts           dq_results
 orders, refunds, ...                         (table, date, metric, value, n, sd)       (value, expected, band,
        │                                                 │                               score, status, reason)
        ▼                                                 ▼                                      ▲
 profile.py ── one GROUP BY per table ──────────► load_series ── contiguous calendar ──► detect.py
   COUNT(*), COUNT(DISTINCT key), SUM(col IS NULL),    missing days → row_count 0        per metric, in date order:
   AVG(x), AVG(x*x), COUNT(x), MAX(loaded_at),                                          expected = L_i + S_i
   per-category counts                                                                  sigma = max(MAD, floor, sampling noise)
   idempotent: re-profiling a range replaces it                                         status ∈ warmup|ok|alert|missing|level_shift
                                                                                                 │
 generate.py ─ synthetic warehouse + ground truth ─► evaluate.py ◄── baselines.py (static rules, 28-day z-score)
```

| module | responsibility |
|---|---|
| `config.py` | table/column config (identifiers validated before they reach SQL), per-metric `MetricSpec` (transform, seasonal, direction, sampling model, `min_delta`), `DetectorParams`, static-rule thresholds |
| `profile.py` | SQL profiler → `dq_metrics` (+ std dev for means) and `dq_category_counts`; fills absent dates with `row_count = 0`; `load_series(until=...)` |
| `detect.py` | sequential detector, PSI with +0.5 smoothing, χ² noise gate, level-shift acceptance, `dq_results` |
| `stats.py` | χ² survival function and inverse (regularized incomplete gamma), normal tail; stdlib only |
| `baselines.py` | static hand-written rules and trailing 28-day z-score, using the same profiled metrics |
| `generate.py` | synthetic `orders`/`refunds` warehouse with seasonality, growth, a benign level shift, a benign mix drift and 7 incident types lasting 1–3 days |
| `evaluate.py` | incident recall, alert precision, false alerts per 100 table-days, per-type recall |
| `cli.py` | `generate`, `profile`, `check`, `series`, `evaluate` |

Metrics per table (configured in [`samples/config.json`](samples/config.json)):

| metric | model | alerts on |
|---|---|---|
| `row_count` | log, seasonal, MAD; `0` with a non-empty history → `missing` | both directions, ≥ ~10% |
| `dup_rate` (needs `key`) | binomial floor | up, ≥ 0.2 points |
| `null_rate:<col>` | binomial floor | up, ≥ 1 point |
| `mean:<col>` | log, standard-error floor | both directions, ≥ ~16% |
| `psi:<col>` | PSI vs pooled last 28 days, χ² gate | up, ≥ 0.10 |
| `freshness_lag_h` (needs `loaded_at`) | hours after partition end | up, ≥ 3 h |

Any spec can be overridden per table and metric with `"overrides": {"null_rate:email": {"min_delta": 0.005}}`.

## Setup

```bash
cd projects/2026-11-12-dqwatch
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # pytest only; the package is stdlib-only
```

## Usage

```bash
# 1. a synthetic warehouse (or point the config at your own SQLite tables)
python -m dqwatch generate --db wh.db --days 120 --seed 7 --truth truth.json --config-out config.json

# 2. profile partitions (all, or --start/--end for an incremental run; re-running is idempotent)
python -m dqwatch profile --db wh.db --config config.json

# 3. check: replays detection, writes dq_results, prints alerts for the latest date (or --date / --since)
python -m dqwatch check --db wh.db --config config.json --since 2025-02-17
python -m dqwatch check --db wh.db --config config.json --date 2025-03-16 --fail-on-alert   # exit 1 → gate a DAG
python -m dqwatch check --db wh.db --config config.json --json

# score one series from CSV (date,value,n) to try settings
python -m dqwatch series --csv samples/null_rate_series.csv --metric null_rate:email [--all]

# benchmark against the baselines
python -m dqwatch evaluate --seeds 11-15 --days 365 --out results --name test_seeds_11-15

# tests
python -m pytest -q
```

In a scheduler, the daily task is `profile --start D --end D` followed by
`check --date D --fail-on-alert`. Downstream jobs run only if the check passes.

### Sample inputs and expected outputs

| input | expected output |
|---|---|
| [`samples/null_rate_series.csv`](samples/null_rate_series.csv): 70 days of an email null rate, ~2% on 800 rows; 2025-04-22 has 3 nulls in only 40 rows (7.5%); 2025-05-02 is a real 9% spike | [`samples/null_rate_expected_alerts.json`](samples/null_rate_expected_alerts.json): only 2025-05-02 alerts. The small-sample day scores 2.3 with the binomial floor; without the floor it alerts (see `test_rate_band_widens_for_small_samples`) |
| `generate --days 120 --seed 7` + [`samples/config.json`](samples/config.json) | [`samples/demo_truth_seed7.json`](samples/demo_truth_seed7.json) (injected incidents), [`samples/demo_check_seed7.txt`](samples/demo_check_seed7.txt) (alerts) |

`tests/test_cli.py` regenerates both and compares them byte for byte.

## Evaluation

The data is synthetic, so the true incidents are known. `orders` has ~1,500 rows/day with
weekday seasonality, ~34%/year growth, a **benign** permanent +35% level shift in the second
half, and a benign PayPal share drift from 25% to 32%. `refunds` has ~60 rows/day, so its
rates and means are noisy. Seven incident types are injected with random magnitudes, some
deliberately subtle, and each lasts 1–3 days:

* volume drop (keep 25–75% of rows)
* missing partition
* duplicate load (2–100% of rows)
* null spike
* unit change (×100 or ÷100)
* category relabelling
* late load (+5–18 h)

How the evaluation is scored:

* An alert counts as **explained** if an incident is active on that table and date and moves
  that metric. Every other alert is **false**. An alert on the benign shift counts as false.
* An incident is **caught** if at least one of its days raises an explained alert.
* Every method sees the same profiled metrics and is scored from day 42 onward.
* The defaults were chosen on seeds 1–3: `freshness min_delta` 3 h, the χ² and standard-error
  floors, and exclusion off. Seeds 11–15 were run once, after those choices were frozen.

**Held-out test seeds 11–15** (365 days each, 3,230 table-days, 225 incidents). Source:
[`results/test_seeds_11-15.md`](results/test_seeds_11-15.md).

| method | incidents caught | recall (min–max per seed) | alerts | precision | false alerts / 100 table-days | alerts after benign +35% shift |
|---|---|---|---|---|---|---|
| **dqwatch** | 208/225 | **0.924** (0.89–0.96) | 353 | **0.915** | **0.93** | 7 |
| dqwatch_exclude_flagged | 207/225 | 0.920 (0.89–0.93) | 361 | 0.903 | 1.08 | 15 |
| dqwatch_no_sampling | 213/225 | 0.947 (0.91–1.00) | 422 | 0.796 | 2.66 | 7 |
| dqwatch_no_seasonal | 197/225 | 0.876 (0.84–0.91) | 372 | 0.790 | 2.41 | 18 |
| zscore_28d | 174/225 | 0.773 (0.75–0.80) | 469 | 0.571 | 6.22 | 3 |
| static_rules | 217/225 | 0.964 (0.95–1.00) | 817 | 0.411 | 14.89 | 34 |

Recall by incident type (caught/injected):

| method | volume_drop | missing_partition | duplicate_load | null_spike | unit_change | mix_shift | late_load |
|---|---|---|---|---|---|---|---|
| dqwatch | 28/36 | 32/32 | 24/26 | 37/38 | 28/28 | 28/34 | 31/31 |
| dqwatch_no_sampling | 28/36 | 32/32 | 26/26 | 37/38 | 28/28 | 31/34 | 31/31 |
| dqwatch_no_seasonal | 17/36 | 32/32 | 24/26 | 37/38 | 28/28 | 28/34 | 31/31 |
| zscore_28d | 6/36 | 30/32 | 25/26 | 35/38 | 26/28 | 25/34 | 27/31 |
| static_rules | 33/36 | 32/32 | 26/26 | 38/38 | 28/28 | 29/34 | 31/31 |

What the numbers say:

* **Static rules catch the most** (+9 incidents), and they pay for it with **16× the false
  alerts** (14.89 vs 0.93 per 100 table-days; precision 0.41). Most of the noise comes from
  week-over-week volume and PSI rules on the 60-row `refunds` table, where ±30% is ordinary
  sampling noise.
* **A trailing z-score is the worst of both.** It misses 30 of 36 volume drops because weekends
  inflate its standard deviation, and it still raises 6.2 false alerts per 100 table-days.
* **Sampling-aware floors are the biggest precision lever.** Without them, false alerts rise
  from 0.93 to 2.66 per 100 table-days. The cost is 5 incidents: 2 small duplicate loads and 3
  category relabellings on the small table that are within sampling noise.
* **Seasonality matters for volume.** Without it, 17 of 36 volume drops are caught instead of
  28, and alerts at the benign shift go from 7 to 18.
* **Excluding flagged points from baselines didn't help.** I expected it to, and it is the
  default in many monitors. With median/MAD baselines, a few bad days already have little
  influence, and exclusion slows adaptation to genuine level shifts (15 vs 7 alerts after the
  benign shift). It is available as `"exclude_anomalies": true` but is off by default. The
  same result appeared on the dev seeds ([`results/dev_seeds_1-3.md`](results/dev_seeds_1-3.md)).
* **16 of the 17 misses are on the 60-row `refunds` table:**
  * 8 volume drops that keep 49–74% of the rows
  * 5 channel relabellings
  * 2 duplicate loads of 2–4% (1–2 rows)
  * a one-day null spike to 28%

  The one `orders` miss relabelled 64% of a 10%-share payment method (PSI below the 0.10
  `min_delta`). Every miss is listed per seed and method under `missed` in
  [`results/test_seeds_11-15.json`](results/test_seeds_11-15.json).

Runtime per 365-day seed on one core: generating ~690k rows takes ~2.2 s, profiling ~0.6 s,
and detection for all six methods ~1.1 s.

## Design decisions

* **Profile once in SQL, detect in Python.** The profile is a single `GROUP BY` per table, and
  each category column adds one more. The detector never touches raw data, so it runs at the
  same speed on any table size, and `dq_metrics` is a cheap, inspectable history.
* **Detection is a pure function of `dq_metrics`.** `check` replays all history and rewrites
  `dq_results` atomically. A re-profiled (reloaded) partition simply changes the inputs, and
  there is no hidden detector state to invalidate.
* **Log transform for counts and means.** Incidents are multiplicative (lose 40%, ×100), and
  weekday effects are too.
* **Decomposition instead of "same weekday last weeks + drift".** The first version used the
  seasonal naive forecast plus a recent-drift correction. A unit test with a +50% step showed
  that it double-counts the shift once the seasonal-naive median catches up, which caused
  false "below expected" alerts two weeks later. Weekday offsets measured against centred
  weekly means don't depend on the level, which removes the problem. The change also cut dev
  false alerts from 0.88 to 0.72 per 100 table-days.
* **Missing partitions are their own status.** `log(0)` is not a score, and an empty
  partition is always worth a page.
* **`level_shift` after 4 consecutive same-direction alerts.** It tells the reader that the
  monitor now treats the new level as normal, so it is reported rather than silently absorbed.
* **One-sided where only one side is bad.** Rates, PSI and freshness alert only upward. Volume
  and means alert both ways.

## Limitations

* Daily partitions only. Hourly data would need intraday seasonality.
* There is no holiday calendar. Black Friday will alert, and acknowledging known events would
  need an annotations table.
* Evaluation uses synthetic data. Real incident mixes, magnitudes and noise differ, and the
  numbers show relative behaviour on this generator, not expected production precision.
* Schema changes (added, dropped or retyped columns), cross-table checks (orders vs refunds
  reconciliation) and distribution drift of numeric columns beyond the mean are not monitored.
* PSI for high-cardinality columns becomes noisy and the χ² approximation weakens. It is meant
  for low-cardinality enums.
* Freshness is measured from `MAX(loaded_at)` in the partition, so a partition that never
  arrives shows up as `missing` volume, not as lag.
* The bundled profiler targets SQLite. Its SQL is plain ANSI aggregates, but other engines
  would need a connection adapter.
