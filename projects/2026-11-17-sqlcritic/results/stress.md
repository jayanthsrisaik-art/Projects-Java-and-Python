# Critic stress test (mutated gold SQL)

Golden examples: {'dev': 176, 'test': 176} (each wording is checked, because the inferred contract depends on wording; counts below are mutant instances over all examples).

False-reject rate on unmodified gold SQL: gold_contract/dev 0.000, gold_contract/test 0.000, inferred_contract/dev 0.000, inferred_contract/test 0.204.

| mutation | kind | effective mutants | recall (gold contract) | recall (inferred contract) | first issue raised |
|---|---|---:|---:|---:|---|
| fanout_join | structural | 64 | 1.000 | 1.000 | fanout_aggregate×16, fanout_count×48 |
| hallucinated_column | structural | 340 | 1.000 | 1.000 | unknown_column×340 |
| destructive_stmt | safety | 352 | 1.000 | 1.000 | not_read_only×352 |
| spurious_group_by | structural | 28 | 1.000 | 1.000 | row_count×28 |
| wrong_year | semantic | 204 | 0.000 | 0.074 | - |
| drop_status_filter | semantic | 196 | 0.000 | 0.046 | - |
| drop_label_column | structural | 84 | 1.000 | 1.000 | column_count×72, unknown_column×12 |
| drop_limit | structural | 48 | 1.000 | 1.000 | row_count×48 |
| drop_distinct | structural | 32 | 1.000 | 1.000 | fanout_count×32 |
