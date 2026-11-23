# gtfsflat benchmark

Seeds: [0, 1, 2, 3, 4]; windows: spring 2026-03-01..2026-03-14, fall 2026-10-25..2026-12-05

## SQL engine vs row-at-a-time oracle

| window | seed | service dates | trip instances | stop events | exact | wrong | missing | extra | SQL build s | oracle s |
|---|---|---|---|---|---|---|---|---|---|---|
| spring | 0 | 29 | 2234 | 20946 | 20946 | 0 | 0 | 0 | 0.097 | 0.084 |
| fall | 0 | 85 | 6549 | 61371 | 61371 | 0 | 0 | 0 | 0.229 | 0.25 |
| spring | 1 | 29 | 2234 | 20946 | 20946 | 0 | 0 | 0 | 0.087 | 0.078 |
| fall | 1 | 85 | 6549 | 61371 | 61371 | 0 | 0 | 0 | 0.223 | 0.297 |
| spring | 2 | 29 | 2234 | 20946 | 20946 | 0 | 0 | 0 | 0.095 | 0.105 |
| fall | 2 | 85 | 6549 | 61371 | 61371 | 0 | 0 | 0 | 0.241 | 0.263 |
| spring | 3 | 29 | 2234 | 20946 | 20946 | 0 | 0 | 0 | 0.125 | 0.131 |
| fall | 3 | 85 | 6549 | 61371 | 61371 | 0 | 0 | 0 | 0.213 | 0.267 |
| spring | 4 | 29 | 2234 | 20946 | 20946 | 0 | 0 | 0 | 0.104 | 0.09 |
| fall | 4 | 85 | 6549 | 61371 | 61371 | 0 | 0 | 0 | 0.26 | 0.285 |

## Shortcut loaders vs correct events (summed over seeds)

| window | variant | expected | exact | wrong time | missing | extra | % events not exact |
|---|---|---|---|---|---|---|---|
| spring | ignore_exceptions | 104730 | 104430 | 0 | 300 | 0 | 0.29% |
| spring | wrap_24h | 104730 | 100072 | 4658 | 0 | 0 | 4.45% |
| spring | wallclock | 104730 | 104476 | 254 (on DST days: 233) | 0 | 0 | 0.24% |
| spring | ignore_frequencies | 104730 | 82530 | 0 | 22200 | 790 | 21.20% |
| spring | naive_all | 104730 | 77489 | 4741 | 22500 | 790 | 26.01% |
| fall | ignore_exceptions | 306855 | 305610 | 0 | 1245 | 7680 | 0.41% |
| fall | wrap_24h | 306855 | 293265 | 13590 | 0 | 0 | 4.43% |
| fall | wallclock | 306855 | 306598 | 257 (on DST days: 154) | 0 | 0 | 0.08% |
| fall | ignore_frequencies | 306855 | 241005 | 0 | 65850 | 2340 | 21.46% |
| fall | naive_all | 306855 | 226166 | 13594 | 67095 | 9300 | 26.30% |

## Departure board: next 3 departures within 60 min (summed over seeds)

| window | variant | night queries wrong (22:00-04:00) | day queries wrong |
|---|---|---|---|
| spring | gtfsflat SQL board (any mismatch vs oracle board, all queries) | 0 / 10000 | |
| spring | board_same_service_date_only | 385 / 2448 (15.7%) | 0 / 7552 (0.0%) |
| spring | ignore_exceptions | 14 / 2448 (0.6%) | 10 / 7552 (0.1%) |
| spring | wrap_24h | 91 / 2448 (3.7%) | 0 / 7552 (0.0%) |
| spring | wallclock | 39 / 2448 (1.6%) | 0 / 7552 (0.0%) |
| spring | ignore_frequencies | 0 / 2448 (0.0%) | 726 / 7552 (9.6%) |
| spring | naive_all | 87 / 2448 (3.6%) | 736 / 7552 (9.7%) |
| fall | gtfsflat SQL board (any mismatch vs oracle board, all queries) | 0 / 10000 | |
| fall | board_same_service_date_only | 376 / 2543 (14.8%) | 0 / 7457 (0.0%) |
| fall | ignore_exceptions | 18 / 2543 (0.7%) | 130 / 7457 (1.7%) |
| fall | wrap_24h | 70 / 2543 (2.8%) | 0 / 7457 (0.0%) |
| fall | wallclock | 14 / 2543 (0.6%) | 0 / 7457 (0.0%) |
| fall | ignore_frequencies | 0 / 2543 (0.0%) | 672 / 7457 (9.0%) |
| fall | naive_all | 77 / 2543 (3.0%) | 790 / 7457 (10.6%) |
