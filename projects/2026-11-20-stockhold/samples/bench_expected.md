# stockhold benchmark

## A. Rush: 200 shoppers, 32 threads, 50 units (mean over 5 runs; timing-dependent)

| strategy | extra check-to-write delay | holds granted | oversold (max) | errors | invariant violations | wall s |
|---|---|---|---|---|---|---|
| immediate | 0 ms | 50.0 | 0.0 (0) | 0 | 0 | 1.4 |
| check-then-act | 0 ms | 61.4 | 11.4 (14) | 0 | 5 | 0.3 |
| immediate | 1 ms | 50.0 | 0.0 (0) | 0 | 0 | 1.7 |
| check-then-act | 1 ms | 69.0 | 19.0 (23) | 0 | 5 | 0.3 |

## B. Lost responses: 80 shoppers, 60 units, limit 2, 30% of responses lost, up to 4 attempts (mean over 5 seeds)

| client | served | refused | gave up | duplicate holds | stranded units | violations |
|---|---|---|---|---|---|---|
| retry with Idempotency-Key | 59.8 | 19.8 | 0.4 | 0.0 | 0.2 | 0 |
| retry without key | 45.4 | 34.2 | 0.4 | 11.4 | 14.6 | 0 |

## C. Abandonment: 250 shoppers over 30 min, 100 units, 40% abandon, payers take 5-180 s (mean over 5 seeds)

| hold TTL | units sold | shoppers refused | payers whose hold expired | violations |
|---|---|---|---|---|
| none | 58.0 | 150.0 | 0.0 | 0 |
| 600 s | 92.8 | 94.6 | 0.0 | 0 |
| 120 s | 96.2 | 17.6 | 46.6 | 0 |
