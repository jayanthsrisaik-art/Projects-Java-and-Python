Seeds [11, 12, 13, 14, 15], 365 days each, 3230 evaluated table-days, 225 injected incidents.

| method | incidents caught | recall (min-max per seed) | alerts | precision | false alerts / 100 table-days | alerts after benign +35% shift |
|---|---|---|---|---|---|---|
| dqwatch | 208/225 | 0.924 (0.89-0.96) | 353 | 0.915 | 0.93 | 7 |
| dqwatch_exclude_flagged | 207/225 | 0.920 (0.89-0.93) | 361 | 0.903 | 1.08 | 15 |
| dqwatch_no_sampling | 213/225 | 0.947 (0.91-1.00) | 422 | 0.796 | 2.66 | 7 |
| dqwatch_no_seasonal | 197/225 | 0.876 (0.84-0.91) | 372 | 0.790 | 2.41 | 18 |
| zscore_28d | 174/225 | 0.773 (0.75-0.80) | 469 | 0.571 | 6.22 | 3 |
| static_rules | 217/225 | 0.964 (0.95-1.00) | 817 | 0.411 | 14.89 | 34 |

Recall by incident type (caught/injected):

| method | volume_drop | missing_partition | duplicate_load | null_spike | unit_change | mix_shift | late_load |
|---|---|---|---|---|---|---|---|
| dqwatch | 28/36 | 32/32 | 24/26 | 37/38 | 28/28 | 28/34 | 31/31 |
| dqwatch_exclude_flagged | 27/36 | 32/32 | 24/26 | 37/38 | 28/28 | 28/34 | 31/31 |
| dqwatch_no_sampling | 28/36 | 32/32 | 26/26 | 37/38 | 28/28 | 31/34 | 31/31 |
| dqwatch_no_seasonal | 17/36 | 32/32 | 24/26 | 37/38 | 28/28 | 28/34 | 31/31 |
| zscore_28d | 6/36 | 30/32 | 25/26 | 35/38 | 26/28 | 25/34 | 27/31 |
| static_rules | 33/36 | 32/32 | 26/26 | 38/38 | 28/28 | 29/34 | 31/31 |
