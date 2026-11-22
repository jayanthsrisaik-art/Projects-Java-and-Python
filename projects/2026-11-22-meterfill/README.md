# meterfill: smart-meter register readings → hourly usage you can bill on

Smart electricity meters don't report hourly consumption. They report a **cumulative register**
(for example `48211.37 kWh`) every 15–60 minutes over a flaky network. What the head-end system
receives is messy:

* **communication outages** of hours to days with no readings at all
* readings that arrive **out of order** and are **re-sent** after retries
* **misreads**: digit errors such as `48211 → 58211`
* **register rollover**: a 5-digit register wraps from `99999.99` to `0.00`
* **meter exchanges**: a new meter starts again from `0.00`

Billing, time-of-use tariffs and demand-response settlement all need **hourly kWh**. The usual
mistakes are to difference consecutive readings naively, which turns a single digit error into
a 10,000 kWh hour, or to "impute" a gap with a forecast and throw away the one thing the meter
actually knows.

The design rests on this point: **when a gap ends with a valid reading, the energy used during the
gap is known exactly, because it is the register difference.** Only its distribution over the
hours is unknown. meterfill keeps that total fixed and uses ML only to decide the hourly
*shape*.

It combines the three disciplines:

| part | what it does |
|---|---|
| **data engineering** | idempotent SQLite ingest (duplicates and conflicting resends), per-meter cleaning that tells a misread from a rollover from a meter exchange, conversion to intervals, a conservation-checked hourly table with a quality flag on every hour |
| **ML** | a global gradient-boosting shape model trained on artificially masked gaps (targets are the fleet's own *actual* hours, no ground truth), a meter-level train/test split, baselines, and evaluation against simulator truth |
| **software** | a FastAPI service (`/readings`, `/weather`, `/rebuild`, `/meters/{id}/usage`, `/meters/{id}/issues`), a CLI, stale-build tracking, and 32 tests |

## Architecture

```
readings.csv / POST /readings      weather.csv / POST /weather
            │                                  │
            ▼                                  ▼
   store.py  SQLite: readings (PK meter_id, ts; first value wins, conflicts logged), weather, meta
            │
            ▼  per meter
   clean.py     sort → classify every reading: first | ok | rollover | misread | reset
                → segments with an unwrapped cumulative value → intervals (t0, t1, kWh)
            │
            ▼
   allocate.py  short interval (≤ 2 h)   → constant rate                    quality = actual
                gap with closing reading → spread ∝ predicted shape, then
                                           rescaled to the register delta   quality = allocated
                no closing reading       → predicted kWh                    quality = estimated
                (before first / after last reading, across a meter exchange)
            ▲
            │ predicted kWh per hour
   model.py / features.py
                uniform  - the meter's mean rate (= linear interpolation of the register)
                profile  - the meter's own hour-of-week mean over its actual hours
                model    - HistGradientBoosting (Poisson loss) on hour, weekday, temperature,
                           heating degrees and the meter's own normalised profile
            │
            ▼
   hourly(meter_id, hour_ts, kwh, quality), reading_flags, issues  →  api.py / cli.py
```

Other files: `simulate.py` (synthetic fleet with labelled faults and true hourly kWh),
`evaluate.py` (multi-seed evaluation), `scripts/make_samples.py` (regenerates `samples/`).

## Setup

Python 3.11+.

```bash
cd projects/2026-11-22-meterfill
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Usage

```bash
# 1. a synthetic fleet: 40 households, 4 weeks (readings.csv, weather.csv, truth_*.csv)
python -m meterfill simulate --out data --households 40 --days 28 --seed 4

# 2. load it (running it again changes nothing: every row counts as a duplicate)
python -m meterfill ingest --db meter.db --readings data/readings.csv --weather data/weather.csv

# 3. train the shape model on the fleet, then build the hourly table
python -m meterfill train --db meter.db --model model.joblib
python -m meterfill build --db meter.db --method model --model model.joblib

# 4. look at one meter
python -m meterfill usage  --db meter.db --meter MTR01003 --granularity day
python -m meterfill issues --db meter.db --meter MTR01003

# 5. or serve it
python -m meterfill serve --db meter.db --model model.joblib --port 8000
```

API:

| endpoint | purpose |
|---|---|
| `POST /readings` `{"readings":[{"meter_id","ts","register_kwh"}]}` | idempotent ingest; returns `inserted / duplicates / conflicts` |
| `POST /weather` `{"hours":[{"hour_ts","temp_c"}]}` | hourly temperatures; these set the hour grid |
| `POST /rebuild?method=uniform\|profile\|model` | re-clean and rebuild every meter. The default is `model` when a model is loaded, otherwise `profile` |
| `GET /meters` | readings, hours by quality and fault count for each meter |
| `GET /meters/{id}/usage?start&end&granularity=hour\|day` | kWh with a quality per hour, or per day with a count of hours by quality |
| `GET /meters/{id}/issues` | misreads, rollovers, resets, gaps (with metered kWh), unmetered spans |

Every response includes `build.stale`. It turns `true` when readings or weather arrive after the
last rebuild.

Smoke run of the commands above (run during development): ingesting the 49,743 rows received gave
48,751 inserted and 992 duplicates. Re-ingesting gave 49,743 duplicates and 0 inserted. The build
covered 40 meters × 672 h: 24,123 actual, 2,664 allocated and 93 estimated hours, with 39 misreads,
2 rollovers and 112 gaps detected. A live `uvicorn` instance accepted a new meter through
`POST /readings`, reported `stale: true` until `POST /rebuild`, then served its hours.

### Samples

`samples/` holds a small fleet with every fault type: 6 meters, 10 days, 2,519 rows in arrival
order with 49 resends, 17 misreads, 2 rollovers and 2 meter exchanges. It contains
`readings.csv`, `weather.csv`, the simulator's `truth_faults.csv` and `truth_hourly.csv`, and
`expected_outputs.json`. That last file holds the ingest counts, build summary, issues per meter
and daily usage, all from the deterministic `profile` method. `tests/test_samples_eval.py`
rebuilds the sample from the CSVs and compares the result with this file. It also checks that the
detected faults equal `truth_faults.csv` exactly.

## Evaluation (measured)

`python -m meterfill evaluate --seeds 7 11 23` (about 50 s) writes `results/report.json` and
`results/REPORT.md`. For each seed it simulates 150 households × 56 days (about 390k readings),
splits meters 70/30 into train and test by a seeded permutation, and trains on the train meters
only. It then deletes readings in up to 4 extra random windows (3–72 h) per test meter, on top of
the natural outages, rebuilds, and scores **every `allocated` hour** against true kWh.

**Gap allocation on held-out meters** (≈7,200 allocated hours and ≈310 gaps per seed; mean [min, max] over 3 seeds):

| method | NMAE ↓ | peak hour within ±1 h ↑ | gap total error |
|---|---|---|---|
| uniform (linear register interpolation) | 0.374 [0.360, 0.386] | 0.143 | 0.001 |
| profile (own hour-of-week mean, rescaled) | 0.261 [0.234, 0.284] | 0.406 | 0.001 |
| **model (GBM shape, rescaled)** | **0.253 [0.228, 0.275]** | **0.435** | 0.001 |
| model_raw (GBM, *not* rescaled; ablation) | 0.262 [0.238, 0.282] | 0.435 | 0.093 |

NMAE = Σ|predicted − true| / Σ true over the scored hours. The peak-hour metric counts calendar
days lying fully inside a gap and checks whether the busiest hour lands within ±1 h of the true
one.

| method | 2–6 h gaps | 6–24 h | 24–80 h |
|---|---|---|---|
| uniform | 0.206 | 0.348 | 0.396 |
| profile | 0.185 | 0.240 | 0.274 |
| model | 0.173 | 0.238 | 0.264 |
| model_raw | 0.228 | 0.259 | 0.266 |

**`estimated` hours** (no closing reading; 172–208 hours per seed): uniform 0.269, profile 0.212,
model 0.185 NMAE.

**Fault detection** (all 150 meters, 3 seeds; 771 misreads, 35 rollovers, 14 meter exchanges):
precision 1.0 and recall 1.0 on every fault type after the fix described below.

**Whole pipeline vs naive differencing** (45 test meters per seed, energy over each meter's
metered span):

| | median span-total error | meters > 1 % off | worst single hour error (median over meters) |
|---|---|---|---|
| naive diff, negatives clipped | 1,468 – 2,016 % | 38–40 / 45 | 20,000 – 28,147 kWh |
| meterfill | 0.0 % | 0 / 45 | 1.7 – 1.8 kWh |

Reading the numbers:

* **Conserving the total is the bigger win; the learned shape adds a smaller gain on top.** On
  short gaps, rescaling cuts the model's error from 0.228 to 0.173 NMAE. Without rescaling, gap
  totals are 9.3 % off. The rescaled methods are 0.1 % off, and that residue comes from partial
  hours at gap edges and the 0.01 kWh register resolution.
* **The model beats the household's own profile by only 0.008 NMAE** (0.253 vs 0.261) and by 3
  points of peak-hour accuracy. Its edge is larger where no total exists (`estimated`: 0.185 vs
  0.212), because there the temperature and level information it learns matters more. This is a
  modest gain, and the README does not claim more.
* **Hourly error stays large in absolute terms** (≈25 % NMAE inside gaps). Much of household
  consumption is random at the hourly scale, such as EV charging nights and appliance spikes, and
  no shape model can recover it from outside the gap.
* The naive pipeline's error is dominated by digit-error misreads. Its size depends on the
  simulator's misread rate (1 in 1,500 readings), which is an assumption, not a measured
  real-world rate.

## Design decisions

* **The register delta is ground truth; ML only shapes it.** Gaps with a closing reading are always
  rescaled to the delta, so a bad model can misplace energy within a gap but cannot create or lose
  any. Only spans with no closing reading use the raw prediction, and they are flagged
  `estimated`.
* **Every hour carries its worst-case provenance** (`actual` < `allocated` < `estimated`), so
  billing can refuse estimated hours and analytics can down-weight allocated ones.
* **Misread, rollover or exchange is decided from neighbours, not single values.** A reading is a
  misread when a later reading agrees with the last accepted one and disagrees with it. A
  rollover needs the previous value above 90,000 kWh, the next below 10,000 kWh, and a plausible
  rate (≤ 25 kW) after unwrapping. A drop that nothing later returns from starts a new segment.
  The energy across that segment break is reported as `unmetered` and estimated, never guessed
  from the raw drop.
* **A fix found during evaluation.** The first version misclassified one meter exchange in seed 11.
  The meter was replaced while its register read 99,916, so the new meter's values looked like a
  wrap 3 hours later. The rule now treats agreement that needs a wrap as ambiguous. A reading
  that agrees with its own successors counts as the start of a new regime, not an outlier. A
  regression test covers this case (`test_exchange_of_a_meter_near_wrap...`). Because the rule
  was tuned after seeing that seed, the detection numbers above are not a held-out estimate.
* **Leakage control.** Household features come only from the meter's *fully actual* hours in the
  current build, so hours inside a gap never describe the household being filled. A test changes
  the consumption inside a masked window and asserts the feature matrix is bit-identical. Training
  targets are the train meters' own actual hours with windows masked in exactly the way inference
  sees gaps. Simulator truth is used only for scoring, and train and test meters never overlap.
* **Global model, scale-free features.** Profiles and targets are divided by the meter's mean rate,
  so one model serves a 0.3 kW flat and a heat-pump house with an EV. A Poisson loss suits the
  non-negative targets.
* **Idempotent, atomic ingest.** `(meter_id, ts)` is the key and the first value received wins. A
  different value for an existing key is logged in `ingest_conflicts` rather than overwriting.
  A bad row rolls back its whole batch.
* **Rebuild, not incremental.** A late reading can turn a gap into actual hours or reclassify a
  suspected misread, so each rebuild recomputes everything in one transaction. That takes seconds
  for a few hundred meters. `stale` tells consumers when a rebuild is due.

## Limitations

* **Synthetic data only.** Household behaviour, fault types and fault rates are simulator
  assumptions. Real fleets have faults not modelled here: clock drift, stuck registers, multiple
  registers per meter, DST-local billing days, and partial-reading payloads. Detection is perfect
  here partly because the simulated faults are clean digit errors.
* **Some cases are undecidable from the register alone.** A meter exchanged at 99,9xx whose new
  values are also plausible as a rollover is read as a rollover, which overstates energy by at
  most 25 kW × the outage length. A misread immediately before a long outage can be misclassified.
  Real systems should join meter-exchange work orders. This project does not.
* **Energy across a meter exchange is estimated**, although the new meter's first reading bounds
  the post-exchange part (it started at 0). Using that bound would need the exchange time.
* **Hours are UTC.** Local-time billing days and DST are out of scope.
* **Single-node SQLite and full rebuilds.** Fine for thousands of meters. Millions would need
  partitioned, incremental rebuilds per meter.
* The model's gain over the household-profile baseline is small (see above). For a deployment, the
  profile method is a reasonable dependency-free default, and the service uses it automatically
  when no model is loaded.
