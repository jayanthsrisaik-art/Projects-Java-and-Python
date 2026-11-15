# clickrank: learning to rank from biased click logs

Search teams want to train a ranker on clicks, which come free in large volume, rather than on
human relevance labels. The problem is that clicks mostly measure **where a result was shown**.
Users look at rank 1 far more often than rank 8. A model trained on raw clicks therefore learns to
copy whatever put items at the top: the current production ranker and its biases.

This project builds the full loop on a synthetic product-search problem whose ground truth is
known, so every claim can be measured:

1. **Data**: messy raw search events (JSONL) go into a deterministic SQLite load. The loader
   handles duplicate deliveries, double clicks, wrong client-side positions, orphan and
   out-of-order clicks, truncated lines and bot users.
2. **ML**: examination propensities are estimated from randomised *swap interventions*. The
   ranker is a pairwise linear model trained with inverse-propensity weights (IPS). It is compared
   with a naive click-trained model, the production ranker, BM25, random order and an oracle
   trained on true labels.
3. **Software**: a FastAPI search service serves the trained model. It logs impressions and clicks
   in exactly the schema the ingest step reads, and it can apply the same swap exploration on a
   small share of traffic.

## Results (measured, `python -m clickrank experiment --seeds 0 1 2`)

Setup: 610 products, 400 queries, 60,000 simulated impressions per seed, 10% of impressions with a
rank 1 ↔ rank k swap. Queries are split 70/30 by query text. Metrics are computed on the
**held-out test queries** (120 per seed) against hidden graded relevance (0–3), reranking the top
20 BM25 candidates.

| method | NDCG@5 mean (min–max over 3 seeds) | NDCG@10 mean (min–max) |
|---|---|---|
| random order | 0.4050 (0.3986–0.4136) | 0.5011 (0.4953–0.5074) |
| BM25 only | 0.6554 (0.6266–0.6897) | 0.7115 (0.6799–0.7354) |
| production ranker (the logger) | 0.4793 (0.4583–0.5070) | 0.5789 (0.5614–0.5983) |
| naive click-trained | 0.7300 (0.7015–0.7469) | 0.7871 (0.7629–0.8005) |
| **IPS, estimated propensities** | **0.8941 (0.8820–0.9155)** | **0.9115 (0.9017–0.9293)** |
| IPS, true propensities | 0.8963 (0.8853–0.9175) | 0.9144 (0.9040–0.9321) |
| oracle (trained on true labels) | 0.9078 (0.9015–0.9199) | 0.9270 (0.9180–0.9394) |

IPS vs naive, NDCG@10 per test query, with a bootstrap 95% CI over queries:

| seed | mean diff | 95% CI | wins / losses / ties |
|---|---|---|---|
| 0 | +0.1315 | [+0.1162, +0.1475] | 117 / 3 / 0 |
| 1 | +0.1388 | [+0.1213, +0.1560] | 113 / 6 / 1 |
| 2 | +0.1031 | [+0.0843, +0.1231] | 100 / 18 / 2 |

What the learned weights show (seed 0): popularity has no effect on true relevance, but the
production ranker boosts it. The naive model learns a `log_popularity` weight of **+1.185**. The
IPS model learns **+0.174**, and the oracle learns −0.008. The naive model also over-weights raw
BM25 (+3.29 vs +1.61), which is the production ranker's other input.

Estimated propensities are close to the truth (seed 0, true curve 1/k):

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| estimated (smoothed) | 1.000 | 0.474 | 0.319 | 0.246 | 0.182 | 0.182 | 0.154 | 0.142 | 0.113 | 0.093 |
| true | 1.000 | 0.500 | 0.333 | 0.250 | 0.200 | 0.167 | 0.143 | 0.125 | 0.111 | 0.100 |

Ingestion (seed 0): 135,818 raw lines. The loader dropped 2,668 duplicate events and 276
malformed lines, collapsed 2,085 repeat clicks, corrected 694 click positions and dropped 498
orphan clicks. It flagged 3 of the 3 simulated bot users with no false positives, removing 599
impressions and 5,984 clicks. The other seeds give similar numbers; see
[`results/report.md`](results/report.md).

Exploration cost: swapped impressions got 1.048–1.095 clicks per impression, against 1.063–1.090
for control impressions. The difference is within noise here because the production ranker is
weak. Against a strong ranker, swaps cost clicks, which is why the swap rate should stay small.

All numbers above come from `results/report.json`, produced by the command shown. The full run
takes about 45 s on a laptop-class CPU.

## Architecture

```
            simulate.py (catalog + users + biased logger + swaps + dirt)      service.py (FastAPI)
                         │                                                        │ /search, /click
                         ▼                                                        ▼
                 raw events JSONL  ◀──────────── same schema ─────────────  service events JSONL
                         │
                ingest.py: dedupe by event_id, keep-earliest by (ts, event_id), validate,
                           join clicks → impressions, recompute positions, filter bots
                         ▼
          SQLite: impressions · shown · clicks · bot_users · ingest_report
                         │
        ┌────────────────┴──────────────────┐
        ▼                                   ▼
 propensity.py                        train.py
 swap estimator p_k/p_1              click pairs (clicked > shown-not-clicked)
 + isotonic (non-increasing)  ─────▶  weight 1 (naive) or 1/p_k (IPS)
                                      pairwise logistic regression (no intercept)
                                              │
                                              ▼
                     model.json ──▶ features.py: BM25 top-20 candidates, 9 observable features
                                              │
                     experiment.py: split by query, evaluate NDCG on hidden grades, report
```

| module | role |
|---|---|
| `catalog.py` | Synthetic catalogue: 10 categories, fictional brands, attribute words and "accessory" items that mention another category's noun (lexical traps). Hidden quality and graded relevance are stored in `truth.json`, separate from the observable `catalog.json`. |
| `text.py` | Tokeniser, query normalisation, BM25 with deterministic tie-breaking. |
| `features.py` | Query parsing against the observable taxonomy; features `bm25, coverage, category_match, brand_match, brand_mismatch, attr_coverage, rating, log_popularity, log_price`. |
| `simulate.py` | Production ranker (BM25 + 0.6 × popularity + per-request jitter), position-based click model (examination `(1/k)^eta`, attractiveness rising with grade), swap interventions, bot users, and the dirty event stream. |
| `ingest.py` | Order-independent full rebuild into SQLite, with a counted report of every cleaning decision. |
| `propensity.py` | Swap-intervention estimator using both swapped documents, then weighted isotonic regression. |
| `train.py` | Pair building, aggregation and fitting; model save/load with a feature-list check. |
| `experiment.py` | End-to-end runs over seeds, paired bootstrap comparison, Markdown/JSON report. |
| `service.py` | FastAPI app: `GET /search`, `POST /click`, `GET /model`, `GET /products/{id}`, `GET /health`. |
| `cli.py` | `experiment`, `simulate`, `ingest`, `train`, `search`, `serve`. |

## Setup

Python 3.10+.

```bash
cd projects/2026-11-15-clickrank
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Full experiment: writes results/report.{md,json}, plus artifacts/ (catalog + IPS model of seed 0)
python -m clickrank experiment --seeds 0 1 2

# Step by step
python -m clickrank simulate --seed 0 --impressions 20000      # artifacts/ + work/events.jsonl
python -m clickrank ingest work/events.jsonl --db work/logs.db  # prints the cleaning report
python -m clickrank train --db work/logs.db --method ipw        # artifacts/model.json
python -m clickrank search "waterproof tent" -k 5

# Serve the model; 5% of searches get a rank-1 <-> rank-k swap for propensity estimation
python -m clickrank serve --explore-rate 0.05 --port 8000
curl "localhost:8000/search?q=kestrel%20wireless%20headphones&k=10&user_id=alice"
curl -X POST localhost:8000/click -H 'content-type: application/json' \
     -d '{"request_id": "<from search>", "product_id": "<one of the results>"}'
python -m clickrank ingest work/service_events.jsonl --db work/service.db
```

Try other settings with `--eta 1.5` (steeper position bias), `--swap-rate 0.02` or
`--impressions 20000`.

### Samples

| file | what it is |
|---|---|
| `samples/raw_events.jsonl` | 40 hand-built raw lines covering every cleaning rule: a retried duplicate, a double click, a wrong position, an orphan click, a click on a product not shown, a click delivered before its impression, an empty query, duplicate result ids, a truncated line, an unknown event type and a bot. |
| `samples/expected_ingest_report.json`, `samples/expected_tables.json` | Exact expected output of ingesting that file. |
| `samples/sample_model.json` | The IPS model trained in the seed-0 experiment. |
| `samples/queries.txt`, `samples/expected_rankings.json` | Top-5 rankings that model produces on the seed-0 catalogue. |

## Tests

```bash
python -m pytest -q        # 21 tests, about 11 s
```

The tests cover:

- BM25, NDCG and relevance-rule unit tests; catalogue determinism; hidden fields never written to
  the observable catalogue.
- Ingest:
  - exact sample output;
  - shuffled input gives identical tables, and re-runs give identical tables;
  - conflicting duplicate event ids resolve deterministically;
  - on simulated data, every dirt type is detected, exactly the true bots are flagged, and
    referential integrity holds.
- Propensity estimator:
  - exact arithmetic on a hand-built log;
  - recovery of the true curve within ±0.06 on a simulated log.
- Training:
  - IPS pair weights are exactly 1/p_k;
  - the pairwise fit recovers a known direction;
  - the split is reproducible and disjoint;
  - on a fresh simulated world, IPS beats naive by more than 0.03 NDCG@10 and drops most of the
    popularity weight.
- Service: search → click → log → ingest round trip; 404/422 validation; exploration swaps rank 1
  with the logged k.
- CLI: simulate → ingest → train → search smoke test, plus the sample rankings.

## Design decisions

- **Ground truth is synthetic on purpose.** Real click logs do not come with true relevance, so
  you cannot check whether de-biasing worked. The simulator keeps hidden quality and grades in a
  separate file. They are used only for evaluation and for simulating users, never as features.
- **Swap interventions rather than a fully randomised ranking.** Swapping rank 1 with one random
  rank k touches only two positions on a small share of traffic. Randomisation makes swapped and
  control impressions comparable, so the click ratio identifies `p_k/p_1` without a click-model
  fit. The estimator combines both swapped documents (a ratio of sums) to reduce variance. Weighted
  isotonic regression then enforces a non-increasing curve, with a floor of 0.02 to cap IPS
  weights at 50.
- **Pairwise IPS (Propensity SVM-Rank style) with logistic loss.** Each click at rank k is
  preferred over every shown-but-not-clicked result, with weight 1/p_k. Naive and IPS models
  share the same features, pairs and regulariser; only the weights differ. Weights are rescaled to
  mean 1, so the L2 strength is comparable across methods. That makes the comparison a clean
  ablation.
- **No click-derived features.** A per-product CTR feature would bring back the position bias and,
  in evaluation, leak clicks from test queries. Features come only from the catalogue and the
  query text.
- **Leakage controls.** The split is by normalised query text, so no test query contributes
  clicks, pairs or propensity estimates. Propensities are estimated from training-query logs only.
  The seed is fixed, so the split is reproducible.
- **Deterministic ingest.** Every "keep one" choice is by `(ts, event_id)` or by canonical JSON,
  never by file position. Replaying an out-of-order or re-delivered stream produces identical
  tables. Click positions are recomputed from the logged result list, because client-reported
  positions are wrong about 1% of the time.
- **Bots are filtered before training.** Bots click everything, which would flatten both the
  propensity curve and the preference pairs. The two rules are ≥4 clicks per impression, or ≥80%
  sub-second click gaps. Both are simple, explainable and stored with a reason in `bot_users`.
- **The service writes the training schema.** Production logging and offline training cannot
  drift apart, and the CLI round trip is tested.

## Limitations

- **The simulated users follow a pure position-based model, the same assumption IPS makes.** Real
  users also show trust bias, cascade behaviour (stopping after a good click), and examination
  that depends on query or device. The results here are an upper bound on what IPS delivers under
  model misspecification.
- **Linear ranker, nine features.** Even the oracle reaches only 0.927 NDCG@10. A GBDT or neural
  ranker would do better, and IPS plugs into either the same way.
- **Reranking only.** Candidates come from BM25 top-20 and are evaluated within that set (rerank
  NDCG). Recall loss in retrieval is not measured.
- **Coverage of the logged ranker.** Items that are never shown in the top 10 get no clicks, so
  IPS cannot learn about them. That is a positivity assumption that small swap exploration only
  partly addresses.
- **Propensities are needed at every logged rank.** The estimator uses impressions with at least
  10 results. A service run with `k < 10` contributes to training but not to propensity
  estimation.
- **Bot rules are heuristics tuned to the simulated bots.** Adversarial or low-rate bots would get
  through.
- **Evaluation is offline only.** There is no counterfactual (IPS) estimate of online CTR and no
  interleaving experiment.
- **Exact sample rankings depend on the stdlib RNG.** The catalogue is regenerated from `random.Random(0)`.
  The sample rankings were produced on Python 3.13; standard-library RNG algorithms are stable
  across recent CPython versions, but this is not checked on older interpreters.
