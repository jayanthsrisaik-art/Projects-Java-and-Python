# asofqa: question answering that cites the policy version in force at the time asked about

Company policies change over time. The parental leave policy has a v1, a v2 that replaced it, a
v3 approved for next January, a draft someone is reviewing, and a proposal that was withdrawn
before it took effect. A RAG assistant indexes all of them. Then it gets asked:

* "How many weeks of parental leave do we get?" The answer must come from v2. The newer v3 is
  approved but not effective yet, and the draft has bigger numbers that BM25 likes just as much.
* "What was it in 2022?" If the policy changed during 2022, the honest answer is both values, each
  with its date range.
* "When did it become 20 weeks?" If 20 weeks only ever appeared in a withdrawn proposal, the right
  answer is "it never did", not that proposal's planned date.
* "What was it in 2019?" There was no policy then, so the system should abstain rather than quote
  something.

**asofqa** is a small, fully local QA pipeline for versioned documents. It parses the time a
question refers to, resolves each policy's **validity intervals** from approved versions only,
answers ranges with every value that applied, abstains when nothing applies, and cites
`[doc, valid_from to valid_to]` for each value. It is measured against two baselines people
actually ship: plain RAG, and RAG that always uses the latest version.

## Results (measured, `python -m asofqa eval`)

The evaluation runs on synthetic corpora with 10 policy families and 2–7 versions each. Accuracy
is exact match: the right number, the full ordered list of values for a range, the right date for
"when" questions, or an abstention where one is due. The **test split** uses unseen seeds
(100–104) *and* question phrasings and policy synonyms that never appear in dev. Those were written
before the first evaluation run, and no rule was tuned on test results (see the honesty note
below).

**Test split: 400 questions per strategy**

| strategy | accuracy | current | point in time | range | relative ("a year ago") | future date | previous | when | abstain recall / precision |
|---|---|---|---|---|---|---|---|---|---|
| `naive` (BM25 top chunk) | 0.140 | 0.04 | 0.17 | 0.16 | 0.02 | 0.10 | 0.32 | 0.14 | 0.00 / – |
| `latest` (newest active, effective version) | 0.593 | 0.94 | 0.56 | 0.36 | 0.80 | 0.64 | 0.00 | 0.88 | 0.33 / 0.54 |
| `asof_nostatus` (ablation: ignores status) | 0.843 | 0.80 | 0.88 | 0.94 | 0.90 | 0.80 | 0.70 | 0.84 | 0.71 / 0.88 |
| **`asof`** (this project) | **0.955** | 0.94 | 0.94 | 0.98 | 0.98 | 0.96 | 0.94 | 0.96 | **1.00** / 0.91 |

Per-seed test accuracy for `asof` was 0.950, 0.975, 0.938, 0.963 and 0.950. On the dev split
(seeds 0–2, 240 questions), `asof` scored 0.983, `latest` 0.567 and `naive` 0.304. The full
report is in [`results/report.md`](results/report.md) and [`results/metrics.json`](results/metrics.json).

**What the errors are** (wrong answers by cause, test split):

* `naive`: 230 *older version* (it quoted a superseded document), 40 wrong policy, 35 wrong
  "when" dates, 21 missed abstentions, 5 quoted unapproved documents. Old and new versions are
  lexically almost identical, so BM25 can't distinguish them.
* `latest`: 107 *newer version*. Every historical question gets today's value. It answers 0% of
  "what was it before the change?" questions correctly, by construction.
* `asof_nostatus`: 32 *unapproved version*. It treats withdrawn proposals and drafts as if they
  took effect. That is the cost of ignoring the `status` field.
* `asof`: 16 *wrong policy* and 2 false abstentions. Every remaining error is a **retrieval routing
  miss** on a test-only synonym. For example, "days off for illness" is routed to Paid Time Off,
  because "days" and "off" outweigh the one "illness" mention. The temporal logic itself made no
  errors on the test split.

**Reader ablation.** Turning off the reader's filter for sentences about the replaced value
("This version replaces the previous figure of 20 days…") changed nothing on either split
(0.955 → 0.955 on test). Those sentences overlap too little with the questions to win. The
filter is kept because it matters when they do overlap
(`tests/test_reader.py::test_history_filter_matters_when_history_is_the_best_lexical_match`),
but this corpus doesn't show it.

## Architecture

```
question ──► temporal.parse(question, ref_date) ──► Intent(point | range | previous | when)
     │                                                      │
     └─► strip temporal phrase ─► BM25 over chunks ─► family vote (top-8 chunk scores per policy)
                                                            │
             corpus.family[policy]  ──►  keep status == active  ──►  validity intervals
             [eff_i, eff_{i+1} - 1 day]   (withdrawn/proposed docs are never "in force")
                                                            │
                       select intervals intersecting the intent (or previous / value search)
                                                            │
                           Reader.read_value(question, doc) per selected version
                         (ExtractiveReader locally, ClaudeReader optional)
                                                            │
                    Answer(values | date | abstain, evidence with doc ids and validity ranges)
```

| module | role |
|---|---|
| `asofqa/corpus.py` | seeded generator: 10 policy families, past/scheduled/withdrawn/proposed versions, Markdown docs with front matter, questions with ground truth (computed from the generator's schedule, not from the pipeline) |
| `asofqa/index.py` | front-matter parsing and validation, paragraph chunks with a title prefix, BM25 (stdlib) |
| `asofqa/temporal.py` | rule-based temporal intent: ISO / US / "March 3, 2022" / "3 March 2022" dates, months, quarters, halves, years, "last/this year/month", "N years/months ago", "previous", "when did … become X" |
| `asofqa/reader.py` | `ExtractiveReader` (picks the value sentence and returns its first unit-bearing quantity), optional `ClaudeReader` |
| `asofqa/pipeline.py` | the `naive`, `latest`, `asof` and `asof_nostatus` strategies and cited answer rendering |
| `asofqa/evaluate.py` | dev/test runs, exact-match scoring, error taxonomy, ablation |
| `asofqa/cli.py` | `generate`, `ask`, `eval` |

## Setup

```bash
cd projects/2026-11-14-asofqa
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pytest only; the package itself is stdlib-only
# optional, for --reader claude:
pip install -r requirements-llm.txt
```

## Usage

```bash
# Ask questions against the bundled sample corpus (seed 7). "Today" is pinned to 2026-11-14.
python -m asofqa ask samples/corpus/docs "What was the paid time off allowance in 2022?"
# Q: What was the paid time off allowance in 2022?
# A: The value changed during that period: 15 [pto-v1, from 2021-08-01 to 2022-04-30]; then 25 [pto-v2, from 2022-05-01 to 2024-04-30].

python -m asofqa ask samples/corpus/docs < samples/questions.txt      # one question per line
python -m asofqa ask samples/corpus/docs --strategy latest < samples/questions.txt
python -m asofqa ask samples/corpus/docs --json "When did the paid time off allowance become 20 days?"
python -m asofqa ask samples/corpus/docs --ref-date 2023-01-01 "What is the remote work allowance?"

# Generate another corpus, or run the full evaluation (about 1 s).
python -m asofqa generate /tmp/corpus --seed 42 --split test
python -m asofqa eval --out results

# Tests
python -m pytest -q
```

Documents you bring need this front matter (validated on load):

```
---
doc_id: pto-v3
policy: pto            # family key: all versions of one policy share it
title: Paid Time Off Policy
version: 3
status: active         # active | proposed | withdrawn
effective: 2024-05-01
---
```

### Sample inputs and expected outputs

* `samples/corpus/`: the seed-7 dev corpus. It contains 36 documents, 80 questions with gold
  answers (`questions.jsonl`), and the generator's ground-truth timeline (`timeline.json`).
* `samples/questions.txt`: 10 hand-picked questions covering every intent type, including a
  value that only a draft proposed and a date before the policy existed.
* `samples/expected_answers_{asof,latest,naive}.txt`: exact outputs of each strategy on those
  questions. `tests/test_cli.py` checks them byte-for-byte, and also checks that the sample
  corpus regenerates identically from its seed.

On that sample, `latest` answers "What was the paid time off allowance on 2022-01-15?" with 30
(today's value), while the correct answer is 15. `naive` says the allowance "became 20 days" on
2027-05-01, quoting a draft that was never approved.

### Optional Claude reader

`--reader claude` swaps the extractive value reader for a Claude call. It uses the Anthropic SDK
with model `claude-opus-5-5`, low effort, and server-side refusal fallbacks enabled
(`fallbacks="default"`). It receives a single version's text, so version selection still happens
in code. Without credentials the CLI exits with a message, and the default extractive mode needs no
network at all. The reader is unit-tested with an injected fake client. **No live API call was made
for this project, so there are no measured results for it.**

## Design decisions

* **Time is resolved in code, not by the reader.** Choosing the version is an interval lookup
  over metadata the corpus already has. The reader (extractive or LLM) only answers "what number
  does *this* document set?". That keeps the hard part deterministic and testable, and the same
  logic works with any reader.
* **Status is part of validity.** Only `active` documents form the timeline. A scheduled change
  is active but in the future, so it counts for "what will it be on 2027-03-01?" and not for
  "what is it now?". Withdrawn proposals and drafts never enter the timeline. The
  `asof_nostatus` ablation measures what skipping this costs (−11.2 points on test).
* **Ranges answer with every value in force**, each with its own interval, instead of picking
  one. "What was it in 2023?" is ambiguous whenever the policy changed that year, so the system
  shows the change rather than hiding it.
* **Abstain explicitly** when the date precedes the first version, or the asked-about value was
  never in force. Abstention recall on test is 1.00 (precision 0.91). The false abstentions all
  come from routing to the wrong policy.
* **The reference date is an input** (`--ref-date`, default 2026-11-14), not the wall clock, so
  "now", "last year" and "a year ago" are reproducible.
* **Retrieval strips the temporal phrase** before BM25, so "in 2023" doesn't pull in documents
  that happen to mention 2023. Policy routing is a vote over the top-8 chunk scores.
* **Baselines are realistic, not strawmen.** `latest` is the usual fix (newest active version
  already in effect) and is right about 94% of the time for "now" questions.

## Honesty notes

* The test phrasings and synonyms were written in the same session as the parser and before the
  first evaluation run. Nothing was tuned on test results. Two reader changes were made after
  inspecting **dev** errors only: allowing "6 paid sick days" as a quantity and penalising
  constraint sentences like "at least 10 days in advance". One chunking bug was fixed after the
  first test run: a heading-only paragraph was being indexed as its own chunk. The `asof` test
  accuracy was 0.955 both before and after that fix. Dev accuracy went from 1.000 to 0.983, and
  the numbers above are from after the fix.
* Dev accuracy near 1.0 is expected rather than impressive. The reader heuristics were built
  against this generator's document style. The test split checks new seeds and new question
  wording, **not new document styles**.

## Limitations

* **Synthetic corpus.** Real policy documents are longer, have more than one number per topic
  (part-time vs full-time, regions), use amendments that change one clause, and often lack clean
  front matter. asofqa expects `status` and `effective` metadata. It does not infer them from prose.
* **Lexical routing is the weak point.** All of `asof`'s test errors come from BM25 picking the
  wrong policy family on synonyms ("days off for illness", "vacation allowance"). Families with
  more versions also get more votes. Dense or hybrid retrieval, or routing on max chunk score
  instead of the sum, would likely help, but neither was tried here.
* **One quantity per policy.** The reader returns a bare number (no unit) and assumes each
  document sets one headline value.
* **The temporal grammar is rule-based English.** It does not handle "before 2023", "since the
  merger", fiscal years, or time zones. When a question has no time expression, it is read as
  "now".
* "When" questions match values exactly after normalisation, so "$2,000" equals "2000" but
  "two thousand dollars" does not.
* The optional Claude reader is untested against the live API.
