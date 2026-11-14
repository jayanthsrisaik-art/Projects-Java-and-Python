# asofqa evaluation

reference date: 2026-11-14

## dev split (seeds 0, 1, 2)

questions per strategy: 240

| strategy | accuracy | policy found | current | point | range | relative | future | previous | when | abstain recall | abstain precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | **0.304** | 0.983 | 0.167 | 0.300 | 0.133 | 0.200 | 0.300 | 0.267 | 0.767 | 0.000 | – |
| latest | **0.567** | 0.983 | 0.967 | 0.483 | 0.333 | 0.700 | 0.633 | 0.000 | 0.933 | 0.316 | 0.750 |
| asof_nostatus | **0.892** | 0.983 | 0.900 | 0.950 | 1.000 | 0.900 | 0.833 | 0.800 | 0.800 | 0.684 | 0.929 |
| asof | **0.983** | 0.983 | 0.967 | 0.983 | 1.000 | 0.967 | 0.967 | 1.000 | 1.000 | 1.000 | 0.950 |

Error breakdown (count of wrong answers by cause):

- **naive**: older_version 87, unapproved_version 28, newer_version 22, missed_abstention 19, incomplete_range 7, wrong_policy 3, wrong_date 1
- **latest**: newer_version 70, missed_abstention 13, older_version 10, incomplete_range 5, wrong_policy 4, false_abstention 2
- **asof_nostatus**: unapproved_version 14, missed_abstention 6, wrong_policy 3, newer_version 2, false_abstention 1
- **asof**: wrong_policy 3, false_abstention 1

Per-seed accuracy:

- **naive**: seed 0: 0.275, seed 1: 0.400, seed 2: 0.237
- **latest**: seed 0: 0.562, seed 1: 0.537, seed 2: 0.600
- **asof_nostatus**: seed 0: 0.838, seed 1: 0.900, seed 2: 0.938
- **asof**: seed 0: 1.000, seed 1: 0.950, seed 2: 1.000

Reader ablation (asof accuracy): asof 0.983, asof (no history filter) 0.983

## test split (seeds 100, 101, 102, 103, 104)

questions per strategy: 400

| strategy | accuracy | policy found | current | point | range | relative | future | previous | when | abstain recall | abstain precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | **0.140** | 0.885 | 0.040 | 0.170 | 0.160 | 0.020 | 0.100 | 0.320 | 0.140 | 0.000 | – |
| latest | **0.593** | 0.953 | 0.940 | 0.560 | 0.360 | 0.800 | 0.640 | 0.000 | 0.880 | 0.333 | 0.538 |
| asof_nostatus | **0.843** | 0.953 | 0.800 | 0.880 | 0.940 | 0.900 | 0.800 | 0.700 | 0.840 | 0.714 | 0.882 |
| asof | **0.955** | 0.953 | 0.940 | 0.940 | 0.980 | 0.980 | 0.960 | 0.940 | 0.960 | 1.000 | 0.913 |

Error breakdown (count of wrong answers by cause):

- **naive**: older_version 230, wrong_policy 40, wrong_date 35, missed_abstention 21, incomplete_range 7, newer_version 6, unapproved_version 5
- **latest**: newer_version 107, older_version 16, wrong_policy 16, missed_abstention 14, false_abstention 6, incomplete_range 4
- **asof_nostatus**: unapproved_version 32, wrong_policy 16, newer_version 7, missed_abstention 6, false_abstention 2
- **asof**: wrong_policy 16, false_abstention 2

Per-seed accuracy:

- **naive**: seed 100: 0.150, seed 101: 0.125, seed 102: 0.175, seed 103: 0.087, seed 104: 0.163
- **latest**: seed 100: 0.625, seed 101: 0.575, seed 102: 0.575, seed 103: 0.550, seed 104: 0.637
- **asof_nostatus**: seed 100: 0.800, seed 101: 0.900, seed 102: 0.838, seed 103: 0.912, seed 104: 0.762
- **asof**: seed 100: 0.950, seed 101: 0.975, seed 102: 0.938, seed 103: 0.963, seed 104: 0.950

Reader ablation (asof accuracy): asof 0.955, asof (no history filter) 0.955
