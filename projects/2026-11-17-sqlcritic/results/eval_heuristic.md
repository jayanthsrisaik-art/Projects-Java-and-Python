# sqlcritic eval - writer=heuristic, split=all

| split | mode | accuracy | wrong answers shown | abstain | precision | 1st-attempt acc | recovered by retry | correct 1st rejected | mean attempts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | single | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 1.00 |
| test | single | 0.114 | 0.886 | 0.000 | 0.114 | 0.114 | 0 | 0 | 1.00 |
| dev | exec_retry | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 1.00 |
| test | exec_retry | 0.114 | 0.886 | 0.000 | 0.114 | 0.114 | 0 | 0 | 1.00 |
| dev | critic | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 1.00 |
| test | critic | 0.131 | 0.869 | 0.000 | 0.131 | 0.114 | 3 | 0 | 1.05 |

Contract inference accuracy (shape and k, from the question alone): dev 1.000 (n=176), test 0.795 (n=176)

## Per family (critic mode): accuracy / wrong answers shown

| family | dev | test |
|---|---:|---:|
| aov_segment | 1.00 / 0 | 0.00 / 6 |
| buyers_of_category | 1.00 / 0 | 0.00 / 10 |
| cancel_share_region | 1.00 / 0 | 0.00 / 8 |
| customers_in_region | 1.00 / 0 | 1.00 / 0 |
| customers_without_orders | 1.00 / 0 | 0.00 / 6 |
| monthly_orders_year | 1.00 / 0 | 0.50 / 3 |
| orders_by_status_year | 1.00 / 0 | 0.50 / 3 |
| orders_in_year | 1.00 / 0 | 1.00 / 0 |
| products_over_price | 1.00 / 0 | 0.00 / 30 |
| revenue_by_region_year | 1.00 / 0 | 0.50 / 3 |
| revenue_category_year | 1.00 / 0 | 0.00 / 30 |
| top_customers_orders | 1.00 / 0 | 0.00 / 6 |
| top_products_revenue_year | 1.00 / 0 | 0.00 / 18 |
| units_category_year | 1.00 / 0 | 0.00 / 30 |
