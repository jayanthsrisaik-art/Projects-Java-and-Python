# ledgerlens: categorise bank transactions from merchants you've never seen

A personal-finance backend with three parts: a **data pipeline** that cleans messy bank descriptors and
loads them into SQLite without duplicates, an **ML classifier** that assigns spending categories and
sends low-confidence rows to a review queue, and a **FastAPI service** where user corrections turn into
merchant rules. It also detects recurring charges (subscriptions, bills, payroll).

## The problem

Raw bank rows look like this:

```
POS DEBIT 0105 TAVOZEN #1318 LAKE CITY FL
DEBIT CARD PURCHASE - TAVOZEN LAKE CITY FL CARD 3888
PAYPAL *MIGUTORZENVO 4127167169
MIGUTORZENVO.COM 871-555-7086 TN
ACH CREDIT CONTOSO LLC PAYROLL 54068009
```

Three things usually go wrong in a tutorial version of this:

1. **Grouping on the raw string.** Every store number, card digit and reference id makes a new
   "merchant". In the synthetic data the 234 merchants show up as **11,126 distinct raw
   descriptors**. Anything keyed on the raw string breaks: rules, per-merchant history and
   subscription detection.
2. **A leaky evaluation.** If you split rows at random, the same merchant ends up in both train and
   test. A lookup table that just memorises merchant names then scores **0.998 macro-F1**. On merchants
   it hasn't seen, the same lookup table scores **0.034**. The thing worth measuring is performance on
   new merchants, so the split has to be grouped by merchant.
3. **Auto-filling every row.** A model that is 77% accurate on new merchants should not silently tag
   every row. Confident predictions should be applied automatically. The rest should go to a review
   queue, and a single correction there should fix every past and future row from that merchant.

## Architecture

```
raw CSV / POST /transactions
        │
        ▼
 normalize.py  ── processor (POS, SQ, TST, PAYPAL, WEB, ACH, …), clean name, merchant_key
        │                 (strip prefixes, store #, card/phone/ref, city+state via gazetteer;
        │                  key = first 16 letters, so truncated and squashed forms agree)
        ▼
 store.py (SQLite) ── idempotent insert on txn_id
        │             category precedence: merchant rule  >  model (conf ≥ threshold)  >  review queue
        │             corrections: update txn (source=user) + upsert merchant rule + re-tag
        │             that merchant's non-user rows; monthly_spend SQL view
        ├── model.py      char 2-4gram + word TF-IDF on clean name, one-hot processor / sign /
        │                 log-amount bucket / price ending → LogisticRegression; abstain threshold
        │                 picked on out-of-fold, merchant-grouped predictions
        ├── recurring.py  per (account, merchant_key, sign): cadence from median gap (weekly,
        │                 biweekly, monthly, yearly) + ≥75% gaps in band + ≥80% amounts within 20%
        └── api.py        FastAPI: /transactions, /corrections, /rules, /summary, /subscriptions
 evaluate.py ── grouped K-fold vs leaky K-fold, baselines, selective accuracy, recurring P/R
 synth.py    ── reproducible synthetic bank exports with hidden ground truth
```

## Setup

```bash
cd projects/2026-11-08-ledgerlens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. No paid services and no network access are needed at runtime.

## Usage

```bash
python -m ledgerlens generate                 # data/transactions.csv (14,860 rows, seed 7)
python -m ledgerlens train                    # artifacts/model.pkl + abstain threshold
python -m ledgerlens evaluate --out reports/metrics.json   # ~1.5 min, 5 grouped folds + 5 leaky folds
python -m ledgerlens demo                     # ingest samples/, re-deliver, apply one correction
python -m ledgerlens ingest --db ledger.db --csv samples/transactions_sample.csv --show
python -m ledgerlens serve --db ledger.db     # http://127.0.0.1:8000/docs
pytest -q                                     # 36 tests
```

API examples:

```bash
curl -X POST localhost:8000/transactions -H 'content-type: application/json' -d '{"transactions":[
  {"txn_id":"x1","account_id":"a1","posted_date":"2026-01-05",
   "descriptor":"TST* LUCKY PIZZERIA 4411","amount":-23.5}]}'
curl 'localhost:8000/transactions?source=review'
curl -X POST localhost:8000/corrections -H 'content-type: application/json' \
     -d '{"txn_id":"x1","category":"dining","apply_to_merchant":true}'
curl 'localhost:8000/summary?month=2026-01&account_id=a1'
curl 'localhost:8000/subscriptions?account_id=a1'
```

## Samples

| file | what it is |
|---|---|
| `samples/transactions_sample.csv` | 155 raw rows for one account, Jan–Mar. Generated with a **different seed** (2026) from the training data, so most merchants are new to the model. Includes one re-delivered duplicate. |
| `samples/transactions_sample_labeled.csv` | the same rows with hidden ground truth |
| `samples/normalization_cases.json` | 14 hand-written descriptor → (processor, name, key) cases |
| `samples/expected_demo_output.json` | output of `python -m ledgerlens demo` (a test checks it is reproduced) |
| `samples/expected_metrics.json` | full output of `python -m ledgerlens evaluate` |

The demo on the sample (threshold 0.8094) gave these results: 154 rows inserted and 1 duplicate
ignored; a second delivery inserted 0. **79 rows were auto-tagged and all 79 match the ground
truth.** 59 rows went to review, and the model's suggestion was right for 44 of them. Accepting one
review item (`RAPIDCAB` → transport) created a rule that re-tagged that merchant's 15 other rows. The
detector found 4 recurring series: payroll (biweekly), one subscription and two utilities (monthly).

## Measured results

Data: `generate(seed=7)` gives 14,701 rows after dedupe, 234 merchants, 60 accounts and 182 days.
Results are from `evaluate(seed=0, folds=5)`, reported as mean ± std over folds. The "true" columns
score against clean labels. Training always uses the noisy user labels (1.5% are wrong).

**Normalisation**

| grouping | keys | keys per merchant | merchants with exactly 1 key | keys shared by >1 merchant |
|---|---|---|---|---|
| raw descriptor | 11,126 | 47.5 | 34 / 234 | 0 |
| `merchant_key` | 234 | 1.0 | 234 / 234 | 0 |

**Categorisation (10 classes)**

| model | grouped K-fold acc (new merchants) | grouped macro-F1 | leaky row K-fold macro-F1 |
|---|---|---|---|
| majority class | 0.213 ± 0.081 | 0.034 ± 0.011 | 0.041 |
| merchant lookup (memorise) | 0.213 ± 0.081 | 0.034 ± 0.011 | **0.998** |
| keyword rules (hand-written) | 0.543 ± 0.050 | 0.500 ± 0.084 | 0.528 |
| **logistic regression** | **0.774 ± 0.081** | **0.762 ± 0.051** | 0.997 |

Pooled per-class F1 for logistic regression on new merchants: income 1.00, subscriptions 0.97,
dining 0.81, utilities 0.80, transport 0.76, shopping 0.76, groceries 0.75, travel 0.73,
health 0.64, fuel 0.61. Fuel and utilities share words such as "ENERGY", and brand-only names like
"TAVOZEN" carry no category word at all.

**Selective prediction (abstain below threshold)**, with the threshold chosen on out-of-fold
training predictions to reach 0.95 precision:

| threshold | coverage (auto-tagged) | accuracy of auto-tagged (true) | accuracy of review-queue suggestions |
|---|---|---|---|
| 0.835 ± 0.061 | 0.560 ± 0.082 | 0.950 ± 0.064 | 0.536 ± 0.140 |

On average the precision target holds on new merchants, but the variation is large: individual folds
missed it. Merchants are clustered, so one hard merchant with hundreds of rows moves a fold's numbers.

**Recurring detection** (322 true recurring series)

| grouping | detected | precision | recall |
|---|---|---|---|
| raw descriptor (baseline) | 95 | 1.000 | 0.295 |
| `merchant_key` | 319 | 1.000 | 0.991 |

The data contains 20 "same store, same weekday, every week" grocery series with variable totals. The
detector flagged **0** of them as recurring. I inspected the 3 recurring series it missed. Two are
$4.99 subscriptions that rose to $5.99: a $1 increase is just over the 20% amount tolerance, so only
4 of 6 amounts count as consistent. The third is a utility series with only 4 charges and one skipped
month, so just 2 of its 3 gaps fall inside the monthly band.

## Design decisions

- **Merchant-grouped CV is the headline metric.** The leaky split is reported next to it to show the
  gap. The test fold is assigned per merchant key and the code asserts there is zero overlap. The
  abstain threshold is chosen inside the training folds (nested grouped CV), never on test rows.
- **Features come only from the row itself**, so training and serving share one code path
  (`row_features`). There are no per-account aggregates that could leak future information.
- **Rule > model > review.** A user's explicit choice is final. A merchant rule outranks the model
  because one correction is strong evidence for every row from that merchant. Re-tagging never
  overwrites a row the user set directly (`source='user'`).
- **Review rows keep `suggested` and `confidence`**, and the monthly summary reports them as
  `needs_review`. Spend is never dropped, so totals always add up (a test checks this).
- **Truncation-tolerant key.** Banks cut names at ~20 characters and PayPal removes spaces. Keying on
  the first 16 letters makes `MAIN STREET OUTFITTE` and `MAINSTREETOUTFITTER` the same merchant.
- **A state code is stripped only after a gazetteer city (or for `.COM` billers)**, so `GOLDEN & CO`
  keeps its `CO`.
- **The amount rule in the recurring detector** is what tells subscriptions apart from habits.
  Duplicate deliveries collapse to one charge per day, so they can't create 0-day gaps.

## Limitations

- **All data is synthetic.** Real descriptors are messier: more processors, international formats,
  merchant renames, pending/posted pairs with different ids. The normaliser is tuned to the
  generator's formats plus the hand-written cases, which is why it scores perfectly here. Expect
  fragmentation and collisions on real exports.
- The city gazetteer is a small hard-coded list. A real system would use a full place-name list.
- The 16-letter key can merge two real merchants that share a long prefix, and it splits a merchant
  whose name the bank truncates below 16 letters in one format but not another.
- Rows from the same merchant can be routed differently (auto vs review) because the amount bucket
  changes confidence. Merchant-level aggregation of confidence would be an improvement.
- Rules are global (one household). Per-account rules, rule deletion and auth are not implemented.
- The model is a single linear classifier with a fixed `C=5`; it was not tuned. Brand-only merchants
  depend on processor and amount signals and are the main error source.
- Models are saved with `pickle`. Only load artifacts you produced yourself.
- Results come from one synthetic world (seed 7). Fold std shows the variance across merchants, not
  across datasets.
