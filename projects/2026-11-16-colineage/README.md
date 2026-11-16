# colineage: column-level lineage for SQL models, checked by perturbing the data

"Can I drop `orders.discount`?" and "why did `vip_customers` change after the status cleanup?"
are column questions. A model DAG (dbt-style table lineage) only answers them at table
granularity: every column of `customer_ltv` "depends on" every column of `orders`. Catalogs
that guess column lineage by matching names miss `id AS order_id` and `quantity * unit_price`.
Most column-lineage tools also only follow **values**. A column that is only used in `WHERE
status = 'complete'`, a join key or a `GROUP BY` still changes downstream numbers when it changes.

colineage parses a project of SQL models and builds a column graph with two kinds of edge:

* **value** edges (`identity` < `transform` < `window` < `aggregate`): data flows into the
  column. Renames, CTE chains, derived tables, `SELECT *`, set operations, window functions and
  scalar subqueries are all followed.
* **control** edges: the upstream column decides which rows exist or how they are grouped,
  matched or deduplicated (`WHERE`, `JOIN ON/USING`, `GROUP BY`, `HAVING`, `DISTINCT`,
  `ORDER BY … LIMIT`, `EXISTS`/`IN` subqueries, `UNION`/`EXCEPT`/`INTERSECT` dedupe).

The analysis is checked in two ways. The output is compared with hand-labelled ground truth,
and a **perturbation oracle** runs every model in SQLite on synthetic data, changes one
source column at a time, and records which model columns actually changed. Every observed
change is a real dependency, so the static lineage must contain all of them.

```
$ python -m colineage impact samples/shop orders.status
orders.status -> 25 column(s) in 7 model(s)
  channel_daily.channel            control
  ...
  stg_orders.status                transform+control
  vip_customers.name               control

$ python -m colineage explain samples/shop vip_customers.name orders.discount
vip_customers.name
  <-[control]- customer_ltv.ltv
  <-[aggregate]- order_totals.net
  <-[transform]- stg_orders.discount
  <-[transform]- orders.discount

$ python -m colineage check samples/shop --drop orders.discount --rename products.supplier=vendor
schema change: drop=['orders.discount'] rename=['products.supplier=vendor']
  breaks 7 model(s), reshapes 0, affects 9 column(s)
  ERROR   contact_list: unknown column 'supplier'
  ERROR   stg_orders: unknown column 'discount'
  BLOCKED order_totals: upstream model(s) failed: stg_orders
  ...
```

(`vip_customers.name` does not carry discount values. It depends on the discount because
`WHERE c.ltv > (SELECT avg(ltv) …)` decides which customers appear. Value-only lineage misses this.)

## Results (measured)

`python -m colineage bench samples/shop samples/saas` (seeds 0, 1, 2; about 2 s). Full output:
[`results/bench.md`](results/bench.md). The unit of measure is an *impact pair* (model column,
source column): "changing this source column can change this model column".

| project | method | precision | recall | F1 | oracle recall |
|---|---|---:|---:|---:|---:|
| shop (10 models, 39 cols), **tuned on** | colineage | 1.000 | 1.000 | 1.000 | 143/143 |
| | value edges only (ablation) | 1.000 | 0.372 | 0.543 | 53/143 |
| | name matching | 0.913 | 0.145 | 0.250 | 21/143 |
| | table-level (model DAG) | 0.368 | 1.000 | 0.538 | 143/143 |
| saas (10 models, 30 cols), **held out** | colineage | **0.943** | **1.000** | **0.971** | 113/113 |
| | value edges only (ablation) | 1.000 | 0.267 | 0.422 | 30/113 |
| | name matching | 0.882 | 0.129 | 0.226 | 15/113 |
| | table-level (model DAG) | 0.423 | 1.000 | 0.595 | 113/113 |

How to read these numbers:

* **shop's perfect score is not a fair estimate.** The first run on shop had 38 false
  positives, all from `LEFT JOIN … GROUP BY <primary key>`. I added the key-preserved rule
  (below) after seeing them. The **saas** project and its labels were written after the analyzer
  was frozen, and it is the honest number. Its 7 false positives come from one pattern: a
  `LEFT JOIN` to a CTE that is unique per join key (`GROUP BY user_id`). colineage does not
  infer uniqueness of derived relations, so it assumes the join may duplicate rows.
* **No false negatives** on either project, against the labels or against the oracle, over 3
  seeds × 4 perturbation types × every source column.
* **The oracle caught 2 labelling mistakes.** In the first saas labels I left out
  `invoices.account_id` for `paid_share.amount` and `paid_share.share` (the `IN (SELECT account_id
  FROM active_users)` filter). The oracle observed both dependencies. The labels were corrected
  and the correction is recorded in [`samples/saas/truth.json`](samples/saas/truth.json) (`_corrections`).
* **Value vs control roles** match the labels on every edge of shop. On saas there is one
  disagreement: `paid_share.share` gets `invoices.currency` as a value edge because a filter
  inside a scalar subquery shapes the value. I labelled it as control.
* Table-level lineage is complete but 58–63% of its pairs are wrong. Name matching is
  mostly right but finds only 13–15% of the real pairs. Dropping control edges loses 63–73% of
  the real impact.

## Architecture

```
 sources.json ─┐                       ┌─► graph.py   trace / impact / explain (value∘control = control)
 models/*.sql ─┴► project.py ─► analyzer.py ─┤
                  topo order,   sqlglot qualify   └─► cli.py check  (re-analyze with dropped/renamed columns)
                  cycles,       + own scope walk
                  blocked       per output column:
                  models        value deps {node: kind}
                                + row control / column control
                                         │
 oracle.py ◄─────────────────────────────┘          evaluate.py
 synthetic rows from sources.json hints (pk, fk,     colineage vs name-match vs table-level,
 values, range, date, null_rate) → SQLite views →    precision/recall vs truth.json,
 perturb one source column (shuffle / resample /     recall vs oracle
 shift / nullify) → diff each column's multiset
```

| module | responsibility |
|---|---|
| `analyzer.py` | parse + `sqlglot.optimizer.qualify` (star expansion, column qualification, validation) then a custom walk: CTEs, derived tables, set ops, correlated subqueries, windows, aggregates, control clauses, key-preserved grouping |
| `project.py` | loads `sources.json` + `models/*.sql`, Kahn topological order, cycle and blocked-model reporting, feeds each model's inferred output schema to its downstream models, applies hypothetical schema changes |
| `graph.py` | end-to-end composition to source columns, impact sets, path explanations; `model.*` row-existence nodes |
| `oracle.py` | seeded synthetic data, SQLite views, PK-respecting perturbations, multiset diff per column; also asserts that SQLite's output columns equal the inferred schema |
| `evaluate.py` | baselines, P/R/F1 on impact pairs, soundness vs oracle |
| `cli.py` | `analyze`, `trace`, `impact`, `explain`, `check`, `verify`, `bench` |

## Setup and usage

Python ≥ 3.10 (tested on 3.11.18).

```bash
cd projects/2026-11-16-colineage
python -m pip install -r requirements.txt

python -m colineage analyze samples/shop --out lineage.json   # per-model + end-to-end lineage
python -m colineage trace   samples/shop customer_rank.country_rank
python -m colineage impact  samples/shop orders.status [--value-only]
python -m colineage explain samples/shop vip_customers.name orders.discount
python -m colineage check   samples/shop --drop orders.discount --rename products.supplier=vendor -v
python -m colineage verify  samples/saas --seeds 0 1 2        # soundness on synthetic data
python -m colineage bench   samples/shop samples/saas         # writes results/
python -m pytest -q
```

Exit codes: `0` ok; `1` `check` found breakage or reshaping, `verify`/`bench` found a missed
dependency, or `explain` found no path; `2` usage error or a model failed to analyze.
`check` is meant for CI: run it with a planned source migration before merging.

A project is a directory with:

* `sources.json`: `{table: {"rows": n, "columns": {col: {"type": INTEGER|REAL|TEXT, ...hints}}}}`.
  Hints are used only by the oracle's data generator: `pk`, `fk: "table.col"`, `values`,
  `range`, `date`, `pattern: name|email`, `null_rate`. `pk` is also used by the analyzer.
* `models/<name>.sql`: one `SELECT` per file (SQLite dialect). Other models are referenced by
  file name, as in dbt without `ref()`.
* optional `truth.json` for `bench`.

### Sample inputs and expected outputs

| path | contents |
|---|---|
| `samples/shop/` | e-commerce: staging, CTE + join + aggregate, LEFT JOIN rollup, `SELECT *`, window rank, `UNION ALL`, correlated `NOT EXISTS` + scalar subquery, `DISTINCT` |
| `samples/saas/` | held-out: `EXCEPT`, `UNION` dedupe, `IN` subquery, `CASE`, `HAVING`, `ORDER BY … LIMIT`, window over aggregate, multi-condition `LEFT JOIN` |
| `samples/*/truth.json` | hand-labelled semantic lineage (definition below) |
| `samples/*/expected/` | golden `lineage.json` and CLI outputs, compared byte for byte by the tests |
| `samples/broken/` | ambiguous column, unknown column, unknown table, cycle, blocked downstream model; `expected_errors.txt` |

## Design decisions

* **Use sqlglot for name resolution.** `qualify()` expands `*`, attaches every column to its
  source alias, rewrites `USING` and `GROUP BY 1`, and rejects unknown columns. I did not write
  my own resolver for that. The lineage walk is separate (`QueryAnalyzer`) because sqlglot's own
  `lineage()` follows values only and works on one column of one query at a time.
* **Sound by default.** If the analyzer cannot prove an edge is irrelevant, it keeps the edge.
  A spurious entry in an impact report costs a reviewer a minute. A missing one ships a broken
  dashboard. The oracle tests this property, and it has a mutation test: with control edges
  removed, the oracle must report misses.
* **Control is mostly relation-wide, with one refinement.** Every column of a query inherits
  the control set of the rows it lives in, including the control of every upstream relation
  (through `model.*` row-existence nodes, so `SELECT count(*) FROM stg_orders` still depends on
  `stg_orders`' filter). The exception is the *key-preserved* rule: `FROM r LEFT JOIN … GROUP BY
  r.<pk>[, other r columns]` with no WHERE, HAVING or DISTINCT yields exactly one row per row
  of `r`. In that case the non-aggregate columns computed from `r` alone get no control from the
  joins or the grouping, and the aggregates keep it. The rule needs a declared `pk`. Tests
  check that it does not fire for inner joins, filters, grouping by joined columns, or grouping
  without the key.
* **Truth means "can change", not "did change".** A label says that changing the source column's
  values *can* change the multiset of the model column's values on some database that respects
  the declared primary keys. The oracle gives a lower bound on the data it generated: 143 of the
  145 shop pairs and 113 of the 116 saas pairs were observed. The rest, such as
  `product_margin.category` depending on `order_items.product_id`, need a category to lose all
  its items, which the generator never produces.
* **Multiset comparison.** The oracle compares each column's values as a multiset, ignoring row
  order. `ORDER BY` without `LIMIT` is therefore not a dependency, and the analyzer agrees.
  Perturbations of a primary key keep it unique (permutations and constant shifts), because the
  truth labels assume keys hold.
* **`check` re-analyzes instead of guessing.** A dropped or renamed column is applied to the
  catalog and the whole project is analyzed again. Models that now fail are `ERROR`, and their
  dependants are `BLOCKED`. `SELECT *` models that silently change shape are `RESHAPE`. Columns
  whose values could change (`-v`) come from the original graph.

## Limitations

* No uniqueness inference for derived relations or for models. This is the source of all 7
  held-out false positives. Only single-column `pk`s declared on source tables are used.
* SQLite dialect only. Recursive CTEs, `VALUES` in `FROM`, table-valued functions and
  `LATERAL` are rejected with an error, not analyzed.
* No Jinja or `ref()` templating. Models reference each other by file name. A CTE that shadows
  a model name is treated as the CTE throughout the file.
* An edge's `kind` is the strongest operation on the path, a coarse label. `window` and
  `aggregate` are ordered arbitrarily (`aggregate` wins).
* A filter inside a scalar subquery in `SELECT` is reported as a value (`transform`) edge, not
  a control edge.
* The evaluation covers two small projects (20 models, 69 columns, 261 labelled pairs) that I
  wrote, labelled by one person. It shows the approach works and where it over-approximates. It
  is not evidence about arbitrary production SQL.
* The oracle can only confirm dependencies that its synthetic data exercises. It can show that
  the analysis is unsound, but it cannot prove the analysis sound.
