Seeds [1, 2, 3], 365 days each, 1938 evaluated table-days, 130 injected incidents.

| method | incidents caught | recall (min-max per seed) | alerts | precision | false alerts / 100 table-days | alerts after benign +35% shift |
|---|---|---|---|---|---|---|
| dqwatch | 120/130 | 0.923 (0.89-0.95) | 209 | 0.933 | 0.72 | 3 |
| dqwatch_exclude_flagged | 119/130 | 0.915 (0.87-0.95) | 215 | 0.902 | 1.08 | 7 |
| dqwatch_no_sampling | 123/130 | 0.946 (0.93-0.98) | 238 | 0.836 | 2.01 | 3 |
| dqwatch_no_seasonal | 116/130 | 0.892 (0.85-0.93) | 241 | 0.763 | 2.94 | 24 |
| zscore_28d | 99/130 | 0.761 (0.73-0.80) | 263 | 0.586 | 5.62 | 1 |
| static_rules | 118/130 | 0.908 (0.87-0.95) | 444 | 0.435 | 12.95 | 20 |

Recall by incident type (caught/injected):

| method | volume_drop | missing_partition | duplicate_load | null_spike | unit_change | mix_shift | late_load |
|---|---|---|---|---|---|---|---|
| dqwatch | 12/15 | 17/17 | 12/14 | 16/16 | 24/24 | 22/27 | 17/17 |
| dqwatch_exclude_flagged | 12/15 | 17/17 | 12/14 | 16/16 | 24/24 | 21/27 | 17/17 |
| dqwatch_no_sampling | 12/15 | 17/17 | 14/14 | 16/16 | 24/24 | 23/27 | 17/17 |
| dqwatch_no_seasonal | 8/15 | 17/17 | 12/14 | 16/16 | 24/24 | 22/27 | 17/17 |
| zscore_28d | 1/15 | 13/17 | 13/14 | 16/16 | 22/24 | 18/27 | 16/17 |
| static_rules | 12/15 | 17/17 | 14/14 | 16/16 | 24/24 | 18/27 | 17/17 |
