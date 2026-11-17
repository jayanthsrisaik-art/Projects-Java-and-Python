# sqlcritic — text-to-SQL with a writer agent, a separate critic, and the eval harness built first

Business users ask questions in plain English, and a model writes SQL. The dangerous failure
isn't a query that errors, because an error is visible. It is a query that runs and returns
a plausible-looking wrong number. **sqlcritic** splits the work between two agents:

* **writer**: turns the question into SQL and runs it in a read-only sandbox
* **critic**: a separate agent that checks the SQL and its *executed result* against the
  shape the question implies (the *result contract*) before anything reaches the user. On
  rejection it sends structured feedback to the writer for another attempt. If the attempt
  budget runs out, the system **abstains** instead of showing an unverified answer.

The golden set and evaluation harness were built **before** the agents. Every number below
is measured on a held-out split, so changes are judged by numbers rather than by feel.

Everything runs locally: a deterministic synthetic SQLite warehouse, an offline heuristic
writer, and an optional live Claude writer.

## Architecture

```
            ┌────────── LangGraph StateGraph ───────────────────────────────────────┐
question ─▶ │ plan ─▶ write ─▶ execute ─▶ critique ─┬─▶ answer   (critic approves)  │
            │ (infer    ▲     (read-only   (static ├─▶ write    (issues + attempts)│
            │ contract) │      sandbox)   runtime  └─▶ abstain  (budget spent)     │
            │           └──── feedback ◀── contract)                               │
            └───────────────────────────────────────────────────────────────────────┘
  eval harness: golden set (dev/test) × arms {single, exec_retry, critic} → accuracy,
  wrong-answers-shown, abstain rate, retries, contract-inference accuracy, regression gate
```

| module | role |
|---|---|
| `db.py` | deterministic synthetic retail warehouse: 4 regions, 400 customers, 60 products, 4,000 orders, 9,915 order lines. Also the schema doc with business definitions and the FK graph |
| `golden.py` | 352 questions in 14 intent families, each with gold SQL and its true result contract. **Wordings 0–1 = dev, 2–3 = test**, so test phrasing is unseen |
| `contract.py` | infers the expected result shape from the question alone: `scalar`, `grouped`, `topk(k)`, `series` or `list`, plus the counted entity |
| `critic.py` | three check layers. **Static** (sqlglot): parses; is a single read-only SELECT; tables and columns exist; no `COUNT`/`SUM`/`AVG` inflated by a one-to-many join (fan-out). **Runtime**: executes within budget. **Contract**: column count, numeric measure, row count, unique group labels, ORDER BY/LIMIT, ordering |
| `writers.py` | `HeuristicWriter` (offline, deterministic) and `ClaudeWriter` (live, Anthropic SDK). A writer sees only the question and the critic's feedback, never the contract or gold SQL |
| `graph.py` | the LangGraph state machine and the three ablation arms |
| `execute.py` | read-only connection (`mode=ro` + `query_only`), VM-step budget, row cap |
| `evaluate.py` | result-based scoring (column-order-free, float-tolerant, tie-aware top-k), metrics, regression gate |
| `stress.py` | mutation test: injects realistic SQL mistakes into gold SQL and measures critic recall and false rejects |

## Setup

```bash
cd projects/2026-11-17-sqlcritic
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 47 tests, ~10 s
```

The warehouse is built on first use at `.cache/warehouse.db` (git-ignored, about 0.5 MB) and is
byte-for-byte reproducible from the seed.

## Usage

```bash
# ask one question (offline heuristic writer, critic on); prints the SQL, rows and attempts trace
python -m sqlcritic ask "Show revenue by region for 2024."

# full evaluation: 3 arms × dev/test → results/eval_heuristic.{md,json}
python -m sqlcritic eval --out-dir results

# CI regression gate: exit 1 if accuracy drops or wrong-answers-shown rises vs a saved report
python -m sqlcritic eval --split dev --modes critic --baseline results/eval_heuristic.json

# critic mutation stress test → results/stress.{md,json}
python -m sqlcritic stress --out-dir results

# score ANY system's SQL ({"id","sql"} per line) against the golden set
python -m sqlcritic score samples/predictions_example.jsonl

# export the full golden set
python -m sqlcritic golden --out golden.jsonl

# live mode (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
python -m sqlcritic ask "Which 5 products brought in the most sales in 2025?" --writer claude
python -m sqlcritic eval --writer claude --split test --out-dir results
```

### Sample inputs and expected outputs (`samples/`)

Tests check these exactly:

* `golden_sample.jsonl`: one dev and one test example per family, with gold SQL and contract.
* `predictions_example.jsonl` → `expected_score.json`: 5 predictions, **2 correct**:
  * correct: gold SQL, and a right answer with its columns in a different order
  * wrong: a fan-out that counts order lines, a revenue query that forgets the
    completed-only rule, and broken SQL
* `expected_ask_dev_wording.json`: answered on the first attempt.
* `expected_ask_heldout_wording.json` (`"Break down 2024 sales across regions."`):
  * The writer's first answer is a single number. The critic rejects it with
    `column_count` (the question asks for one row per region).
  * The writer then returns a per-region table, but of **order counts, not revenue**,
    because "sales" isn't in its dev vocabulary.
  * The critic passes it: the shape is right and the semantics are wrong. This is the
    limit the stress test quantifies below.

## Results (measured)

### 1. Eval harness: offline heuristic writer, three arms (`results/eval_heuristic.md`)

| split | arm | accuracy | wrong answers shown | abstain | 1st-attempt acc | recovered by retry |
|---|---|---:|---:|---:|---:|---:|
| dev (176) | single | 1.000 | 0.000 | 0.000 | 1.000 | 0 |
| test (176) | single | **0.114** | 0.886 | 0.000 | 0.114 | 0 |
| test (176) | exec_retry | 0.114 | 0.886 | 0.000 | 0.114 | 0 |
| test (176) | critic | **0.131** | 0.869 | 0.000 | 0.114 | 3 |

**The writer scores 100% on dev wording and 11% on held-out test wording.** This is the
trap the eval-first approach exists to catch. Every rule in the heuristic writer was written
while looking at dev phrasing, so judging it on those questions says nothing about real
users. Without a separate test split, the dev number would have looked like a finished
product.

The critic recovers 3 answers on test (11.4% → 13.1%) through shape feedback. It cannot fix
the rest: nearly all of the writer's test errors use the wrong measure or drop a filter
while still returning a well-formed result. The executor-retry arm adds nothing because the
heuristic writer never emits SQL that fails to run. These numbers describe a keyword-based
baseline, not an LLM writer. The live Claude writer is implemented and unit-tested against a
fake client, but it was **not run** here (no API credentials in this environment), so no LLM
accuracy is claimed.

Contract inference (the critic's own reading of the question) is 1.000 on dev wording and
**0.795 on test wording**.

### 2. Critic stress test: realistic mistakes injected into gold SQL (`results/stress.md`)

Only *effective* mutants are counted: their result differs from gold, or they fail. All 352
examples are used.

| mutation | kind | mutants | recall (gold contract) | recall (inferred contract) |
|---|---|---:|---:|---:|
| fan-out join (`COUNT(*)` / `AVG` over orders ⋈ order_items) | structural | 64 | 1.000 | 1.000 |
| `COUNT(DISTINCT x)` → `COUNT(x)` | structural | 32 | 1.000 | 1.000 |
| hallucinated column | structural | 340 | 1.000 | 1.000 |
| dropped label column | structural | 84 | 1.000 | 1.000 |
| dropped `LIMIT` on top-k | structural | 48 | 1.000 | 1.000 |
| spurious `GROUP BY` on a scalar | structural | 28 | 1.000 | 1.000 |
| appended `DROP TABLE` | safety | 352 | 1.000 | 1.000 |
| wrong year literal | semantic | 204 | **0.000** | 0.074* |
| dropped `status = 'completed'` rule | semantic | 196 | **0.000** | 0.046* |

False rejects on unmodified gold SQL:

* **0.000** with the gold contract
* with the inferred contract: **0.000** on dev wording and **0.204** on test wording

\* These non-zero semantic "recalls" are not real detections. They are the same test-wording
contract misreads that cause the 20.4% false rejects.

**Takeaways**

* The critic reliably catches structural, schema and safety mistakes. These include the
  quietly wrong ones that still run, such as fan-out double counting and a missing LIMIT.
* It **cannot** catch a wrong filter value or a broken business rule, because the result
  still has the expected shape. That needs semantic checks: per-question unit tests, value
  ranges, or reconciliation against a metrics layer.
* The critic is only as good as its contract inference. A rule-based contract on unfamiliar
  phrasing rejects 1 in 5 correct answers, so in production the contract should come from
  an LLM or from a curated question-type catalogue, and be evaluated on its own (the harness
  reports this separately).

## Design decisions

* **Eval first, split by wording.** Splitting by wording rather than by random rows means
  memorised phrasing can't inflate test scores. The harness scores *executed results*: column
  order doesn't matter, floats are compared at 4 decimals, and ties at the top-k cut-off are
  accepted. So correct SQL that is written differently isn't marked wrong.
* **"Wrong answers shown" is the headline next to accuracy.** For a BI assistant a wrong
  number shown to a user costs more than an abstention, so the critic gates the answer rather
  than annotating it.
* **The critic doesn't share the writer's view.** It never sees the writer's interpretation,
  only the question, the SQL and the result. The writer never sees the contract, only the
  critic's feedback. This keeps the two agents' failure modes independent.
* **Fan-out detection uses the FK graph.** A `COUNT`/`SUM`/`AVG` over a table joined to one
  of its one-to-many descendants is flagged unless it is `DISTINCT`. This is the classic
  silently-wrong text-to-SQL bug.
* **Sandboxing:** a `mode=ro` URI plus `PRAGMA query_only`, a VM-instruction budget through
  SQLite's progress handler, and a row cap. Static checks also reject multi-statement and
  non-SELECT SQL before execution.
* **Live mode:**
  * calls `claude-opus-5-5` through the official `anthropic` SDK
  * effort set explicitly (`medium`, configurable)
  * server-side refusal fallback enabled (`fallbacks="default"`); turn it off by removing
    the two lines in `ClaudeWriter.write`
  * `refusal` and `max_tokens` stop reasons become writer errors, which make the graph abstain
* **LangGraph** gives an explicit, inspectable state machine. AutoGen-style free-form agent
  chat was deliberately not used: the loop is short, bounded and needs to be reproducible
  for evaluation.

## Limitations

* The data and questions are synthetic: 14 intent families, one 5-table schema. The test
  split measures new wording for *known* intents, not new intents or new schemas.
* The author wrote both the golden set and the heuristic/contract rules. The rules were
  restricted to dev wording, but that is a discipline, not a guarantee. A golden set written
  by other people (or from real query logs) would be a stronger test.
* The offline writer is a keyword baseline for exercising the loop. Its accuracy says nothing
  about LLM writers, and no LLM results are reported because live mode wasn't run here.
* The critic only checks shape and structure (see the stress test). It doesn't detect wrong
  filters, wrong business rules or wrong time windows.
* Contract inference is rule-based and is the critic's weakest link on unfamiliar phrasing.
* SQLite only. Fan-out analysis covers aggregates in each SELECT scope and doesn't reason
  through window functions or correlated subqueries.
