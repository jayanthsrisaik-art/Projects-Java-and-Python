# Ordering strategies, one column (seed 7)

| workload | strategy | rows written / op | renumbers | mean key chars (final) | max key chars (final) | precision lost at op | order correct |
|---|---|---:|---:|---:|---:|---:|---|
| append (1000) | dense integer index | 1.00 | 0 | int | int | - | yes |
| append (1000) | gapped integers (1024) | 1.00 | 0 | int | int | - | yes |
| append (1000) | float64 midpoint | 1.00 | 0 | 8 bytes | 8 bytes | - | yes |
| append (1000) | fraction-only keys | 1.00 | 0 | 100.5 | 200 | - | yes |
| append (1000) | rankboard keys | 1.00 | 0 | 2.9 | 3 | - | yes |
| append (1000) | rankboard keys + rebalance@48 | 1.00 | 0 | 2.9 | 3 | - | yes |
| prepend (1000) | dense integer index | 500.50 | 0 | int | int | - | yes |
| prepend (1000) | gapped integers (1024) | 1.00 | 0 | int | int | - | yes |
| prepend (1000) | float64 midpoint | 1.00 | 0 | 8 bytes | 8 bytes | - | yes |
| prepend (1000) | fraction-only keys | 1.00 | 0 | 83.8 | 167 | - | yes |
| prepend (1000) | rankboard keys | 1.00 | 0 | 2.9 | 3 | - | yes |
| prepend (1000) | rankboard keys + rebalance@48 | 1.00 | 0 | 2.9 | 3 | - | yes |
| hotspot (1000) | dense integer index | 499.50 | 0 | int | int | - | yes |
| hotspot (1000) | gapped integers (1024) | 46.13 | 90 | int | int | - | yes |
| hotspot (1000) | float64 midpoint | 1.00 | 0 | 8 bytes | 8 bytes | 55 | NO |
| hotspot (1000) | fraction-only keys | 1.00 | 0 | 83.8 | 167 | - | yes |
| hotspot (1000) | rankboard keys | 1.00 | 0 | 85.5 | 169 | - | yes |
| hotspot (1000) | rankboard keys + rebalance@48 | 2.67 | 3 | 5.2 | 30 | - | yes |
| mixed (2000) | dense integer index | 144.68 | 0 | int | int | - | yes |
| mixed (2000) | gapped integers (1024) | 1.45 | 1 | int | int | - | yes |
| mixed (2000) | float64 midpoint | 1.00 | 0 | 8 bytes | 8 bytes | - | yes |
| mixed (2000) | fraction-only keys | 1.00 | 0 | 95.2 | 226 | - | yes |
| mixed (2000) | rankboard keys | 1.00 | 0 | 3.3 | 4 | - | yes |
| mixed (2000) | rankboard keys + rebalance@48 | 1.00 | 0 | 3.3 | 4 | - | yes |
