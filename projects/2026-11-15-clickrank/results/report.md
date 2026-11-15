# clickrank results

Seeds: [0, 1, 2]; 60000 impressions per seed, swap rate 0.1, true eta 1.0. Held-out test queries per seed: [120, 120, 120].

## Test-query NDCG (reranking the 20 BM25 candidates, graded ground truth)

| method | NDCG@5 mean (min-max) | NDCG@10 mean (min-max) |
|---|---|---|
| random | 0.4050 (0.3986-0.4136) | 0.5011 (0.4953-0.5074) |
| bm25 | 0.6554 (0.6266-0.6897) | 0.7115 (0.6799-0.7354) |
| production | 0.4793 (0.4583-0.5070) | 0.5789 (0.5614-0.5983) |
| naive_clicks | 0.7300 (0.7015-0.7469) | 0.7871 (0.7629-0.8005) |
| ipw_estimated | 0.8941 (0.8820-0.9155) | 0.9115 (0.9017-0.9293) |
| ipw_true | 0.8963 (0.8853-0.9175) | 0.9144 (0.9040-0.9321) |
| oracle_labels | 0.9078 (0.9015-0.9199) | 0.9270 (0.9180-0.9394) |

## IPS vs naive, per seed (NDCG@10 on test queries, bootstrap 95% CI over queries)

| seed | mean diff | 95% CI | wins / losses / ties |
|---|---|---|---|
| 0 | +0.1315 | [+0.1162, +0.1475] | 117 / 3 / 0 |
| 1 | +0.1388 | [+0.1213, +0.1560] | 113 / 6 / 1 |
| 2 | +0.1031 | [+0.0843, +0.1231] | 100 / 18 / 2 |

## Estimated vs true examination propensity (seed 0)

| rank | raw estimate | smoothed | true |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 0.474 | 0.474 | 0.500 |
| 3 | 0.319 | 0.319 | 0.333 |
| 4 | 0.246 | 0.246 | 0.250 |
| 5 | 0.178 | 0.182 | 0.200 |
| 6 | 0.185 | 0.182 | 0.167 |
| 7 | 0.154 | 0.154 | 0.143 |
| 8 | 0.142 | 0.142 | 0.125 |
| 9 | 0.113 | 0.113 | 0.111 |
| 10 | 0.093 | 0.093 | 0.100 |

## Learned weights (seed 0)

| feature | bm25 | production | naive_clicks | ipw_estimated | ipw_true | oracle_labels |
|---|---|---|---|---|---|---|
| bm25 | +1.000 | +1.000 | +3.292 | +1.606 | +1.612 | +2.137 |
| coverage | +0.000 | +0.000 | -0.003 | +0.355 | +0.298 | +5.919 |
| category_match | +0.000 | +0.000 | +1.911 | +1.883 | +1.878 | +5.911 |
| brand_match | +0.000 | +0.000 | +0.575 | +0.507 | +0.508 | +1.361 |
| brand_mismatch | +0.000 | +0.000 | -0.575 | -0.507 | -0.508 | -1.361 |
| attr_coverage | +0.000 | +0.000 | +0.623 | +0.476 | +0.479 | +0.780 |
| rating | +0.000 | +0.000 | +0.941 | +0.936 | +0.936 | +3.473 |
| log_popularity | +0.000 | +0.600 | +1.185 | +0.174 | +0.146 | -0.008 |
| log_price | +0.000 | +0.000 | -0.027 | -0.019 | -0.022 | -0.017 |

## Ingestion and bot filtering

| seed | bot_clicks_removed | bot_impressions_removed | bot_users | click_positions_corrected | clicks_loaded | duplicate_events | impressions_loaded | lines | malformed_lines | orphan_clicks | repeat_clicks_collapsed | bots flagged / true / TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5984 | 599 | 3 | 694 | 64435 | 2668 | 59273 | 135818 | 276 | 498 | 2085 | 3 / 3 / 3 |
| 1 | 6247 | 626 | 3 | 658 | 62906 | 2650 | 59244 | 134528 | 261 | 542 | 2052 | 3 / 3 / 3 |
| 2 | 5742 | 575 | 3 | 648 | 64527 | 2658 | 59313 | 135595 | 268 | 473 | 2039 | 3 / 3 / 3 |

## Cost of the swap interventions (clicks per impression)

| seed | control | swapped |
|---|---|---|
| 0 | 1.0862 | 1.0950 |
| 1 | 1.0633 | 1.0482 |
| 2 | 1.0902 | 1.0675 |
