# safeshare — shareable, k-anonymous extracts from multi-table data

Data engineering · 2026-11-19 · Python 3.11+ standard library (pytest for tests)

## Problem

A clinic (or any team holding personal data) wants to hand an extract of
`patients` + `encounters` to analysts, a vendor or a research partner. The
usual approach — "drop name, email and phone, keep the rest" — is not
anonymisation. Date of birth, sex and zip code together single out almost
everyone, and anyone with a public roll (voter file, social profile) can link
them back to names. On this project's synthetic data, **99.1% of
roll-listed patients are uniquely re-identified, and 99.3% have their
diagnosis disclosed**, after the direct identifiers are dropped.

`safeshare` is a release pipeline that turns a declared policy into an
extract you can defend, and then **attacks its own output** to measure what
still leaks and how much analytic utility is left.

## What it does

```
input/*.csv ─► fail-closed policy check ─► PII scan of verbatim columns
           ─► k-anonymity / l-diversity on the subject table's quasi-identifiers
           ─► drop suppressed subjects from *every* table
           ─► keyed tokens · per-subject date shift · drop identifiers
           ─► sort rows · re-parse written cells and re-verify k / l
           ─► atomic swap into out/      (+ release_report.json)

safeshare evaluate:  prosecutor linkage attack with a public roll
                     + random count-query workload  ->  privacy / utility numbers
```

* **Fail-closed policy** (`samples/policy.json`). Every column of every table
  needs an explicit action: `drop`, `token`, `date_shift`, `qi`, `sensitive`
  or `keep`. A new upstream column (`ssn`), a missing column, or an
  undeclared table file aborts the release. Nothing gets through by default.
* **Keyed pseudonyms.** `token` = HMAC-SHA256(key, domain, value). The same
  patient gets the same token in `patients` and `encounters`, so joins still
  work. Different domains get unrelated tokens. Without the key the tokens
  can't be reversed by hashing every plausible id, which is what breaks a
  plain unkeyed hash.
* **Per-subject date shifting.** Each patient's encounter dates move by one
  secret offset in ±`max_date_shift_days` (never 0). Intervals within a
  patient ("readmitted after 6 days") stay exact; absolute visit dates are
  hidden.
* **PII scan** of the columns that would be copied verbatim. It looks for
  e-mail, phone and SSN patterns, plus *known identifiers*: any value or word
  from a dropped or tokenised identifier-like column, such as a patient's
  first name inside a free-text note. Findings print masked fragments only
  (`6**********2`), never the value. A column can be acknowledged with
  `"allow_pii": true`.
* **Anonymisation of the quasi-identifiers** (`dob` date, `sex`
  categorical, `zip`), in three modes:
  * `mondrian` (default): strict multidimensional Mondrian. It cuts the
    widest normalised dimension at the allowable point closest to the median.
    A cut is allowable when both halves keep ≥ k rows and ≥ l distinct
    diagnoses. It releases each partition's **cut region**, whose inner
    boundaries fall midway between observed values and whose outer bounds are
    padded to year / 3-digit-zip blocks. No suppression.
  * `mondrian_tight`: the same partitions released as min/max bounding boxes.
    This is the textbook variant, kept as an ablation (see results).
  * `full_domain` (baseline): one generalisation level per column for the
    whole table (e.g. 5-year birth blocks + 3-digit zip). It is chosen by
    exhaustive search of the hierarchy lattice for minimum information loss,
    and suppresses failing classes up to `suppression_limit` (Samarati /
    Incognito style).
  * `none` (baseline): direct identifiers dropped, QIs released exactly.
* **Self-verification.** After writing, the subject CSV is parsed back from
  text and k / l are recomputed from the released cells. A mismatch aborts
  the release, and the previous output directory stays untouched.

## Results (measured)

`python -m safeshare bench` (5 seeds × 5,000 synthetic patients, ~14k
encounters, a public roll that covers 70% of patients plus 2,500
non-patients; 3m26s on this machine). Full table: [`results/bench.md`](results/bench.md),
raw data per run: [`results/bench.json`](results/bench.json).

| method | unique re-id | mean risk | max risk | diagnosis disclosed | DOB endpoint exposure | avg NCP | suppressed | query err median / p90 |
|---|---|---|---|---|---|---|---|---|
| drop identifiers only | 0.991 | 0.996 | 1.000 | 0.993 | 0.000 | 0.000 | 0 | 0.000 / 0.000 |
| full-domain k=5 | 0.000 | 0.024 | 0.200 | 0.000 | 0.063 | 0.065 | 18.2 | 0.259 / 0.975 |
| **mondrian k=5** | 0.000 | 0.153 | 0.200 | 0.001 | 0.039 | 0.033 | 0 | 0.094 / 0.427 |
| mondrian k=5, tight boxes | 0.000 | 0.153 | 0.200 | 0.001 | **1.000** | 0.017 | 0 | 0.067 / 0.297 |
| full-domain k=5 l=3 | 0.000 | 0.024 | 0.200 | 0.000 | 0.063 | 0.065 | 19.4 | 0.259 / 0.975 |
| **mondrian k=5 l=3** | 0.000 | 0.150 | 0.200 | **0.000** | 0.039 | 0.033 | 0 | 0.094 / 0.432 |
| mondrian k=10 l=3 | 0.000 | 0.073 | 0.100 | 0.000 | 0.043 | 0.047 | 0 | 0.125 / 0.596 |

What the numbers say:

* **Dropping identifiers alone fails.** 99.1% of targets are uniquely
  re-identified. Every k-anonymous release caps the per-target risk at 1/k
  (max risk 0.200 for k=5, 0.100 for k=10), and nothing is re-identified
  uniquely.
* **Mondrian keeps far more utility than one-size-fits-all generalisation.**
  At the same k=5 it has half the information loss (NCP 0.033 vs 0.065) and a
  2.8× lower median count-query error (0.094 vs 0.259; p90 0.43 vs 0.98). It
  also suppresses nobody, where full-domain drops 18 patients per run.
* **The price is that Mondrian sits closer to the bound.** Its classes have
  a median size of 6 against 38 for full-domain, so the *mean* prosecutor
  risk is higher (0.153 vs 0.024). Both meet the same k guarantee. If the
  mean matters to you, raise k: k=10 halves it to 0.073 and still beats
  full-domain k=5 on query error (0.125 vs 0.259).
* **k-anonymity alone still leaks diagnoses sometimes.** Mondrian k=5
  produced classes with a single diagnosis, and 0.1% of targets had theirs
  disclosed with certainty. Adding l=3 brings that to exactly 0.000 at
  essentially no utility cost. The full-domain classes were large enough to
  be 3-diverse already.
* **Textbook tight bounding boxes publish real birth dates.** In every
  `mondrian_tight` class, the DOB range starts or ends on a member's exact
  birth date (exposure 1.000). Region cells bring that to 0.039. That residue
  comes from adjacent-day neighbours and sits below the 0.063 coincidence rate
  of the full-domain grid, whose blocks just happen to start on someone's
  birthday. The narrower boxes cost about 0.03 in query error.

## Setup

```bash
cd projects/2026-11-19-safeshare
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # only pytest; runtime is stdlib
python -m pytest -q                      # 102 tests, ~3 s
```

## Usage

```bash
# 1. synthetic data (writes work/demo/clinic/*.csv and work/demo/clinic_public/voters.csv)
python -m safeshare synth --out work/demo/clinic --patients 2000 --seed 0

# 2. scan only: see what the leaky policy (notes kept verbatim) would publish
python -m safeshare scan --input work/demo/clinic --policy samples/policy_leaky_notes.json
#    -> phone + known_identifier findings in encounters.notes, exit code 3

# 3. release (use --key-file or $SAFESHARE_KEY for real data)
python -m safeshare release --input work/demo/clinic --policy samples/policy.json \
    --out work/demo/release --method mondrian --k 5 --l 3 --demo-key

# 4. attack + utility evaluation of that release
python -m safeshare evaluate --original work/demo/clinic --release work/demo/release \
    --voters work/demo/clinic_public/voters.csv --policy samples/policy.json --demo-key

# 5. full benchmark -> results/bench.md, results/bench.json
python -m safeshare bench --seeds 5 --patients 5000
```

Exit codes: `0` ok · `2` policy/input/key error · `3` PII found · `4` k/l
infeasible.

### Sample input and expected output

* `samples/clinic/` – 300 synthetic patients, 842 encounters (seed 42);
  `samples/clinic_public/voters.csv` – the attacker's roll.
* `samples/expected/release/` – the exact output of
  `release --input samples/clinic --policy samples/policy.json --demo-key`
  (Mondrian k=5: 48 classes, verified k=5, l=2, NCP 0.1365).
* `samples/expected/evaluate.json` – its evaluation (0 unique re-ids, max
  risk 0.2, median query error 0.140).

Both are checked byte for byte (and value for value) by `tests/test_cli.py`.

A released patient row looks like:

```
patient_id,dob,sex,zip,plan,diagnosis
pat_22anxhlo5jlvkqx2,1956-04-25..1965-12-28,F,01791,medicare,anxiety
pat_22hxhe5hnlp7g33x,2006-01-19..2007-01-17,F,01946-01956,silver,diabetes
```

## Tests (102)

* `test_qi.py`: format/parse round-trips for every cell kind, zip prefix
  notation, negative numeric ranges, nested hierarchy levels, NCP bounds.
* `test_anonymize.py`: property tests over random data (4 seeds × k∈{2,5,12}
  × l∈{1,3} × tight/region). Every class has ≥ k rows and ≥ l diagnoses,
  and every cell contains the record's true value. Region cells partition
  the space, and a tight box always sits inside its region. Also covers the
  full-domain suppression budget, infeasibility and the identity baseline.
* `test_pseudonym.py`: token stability, domain separation, key dependence,
  no collisions on 20k ids, bounded non-zero shifts that preserve intervals,
  time-of-day suffixes, key loading rules.
* `test_policy.py`: ten invalid-policy cases, plus fail-closed on a new
  `ssn` column, a missing column and an undeclared table file.
* `test_piiscan.py`: leaky notes are caught with masked examples only, the
  release refuses and leaves no output or temp dir, `allow_pii`, and no false
  positives from dropped free text.
* `test_release.py`: no raw id/name/email/phone anywhere in the output,
  referential integrity, one constant non-zero shift per patient,
  deterministic output, key-dependent tokens, suppressed subjects removed
  from child tables, failed releases keep the previous output, atomic
  replacement.
* `test_evaluate.py`: the identifier-drop baseline is re-identifiable,
  k-anonymous releases cap risk at 1/k, l-diversity gives 0 disclosure, tight
  boxes expose birth dates, and Mondrian beats full-domain on NCP and query
  error.
* `test_cli.py`: all exit codes, plus sample outputs matching
  `samples/expected/`.

## Design decisions

* **Fail-closed over convenient.** Schema drift is the most common way
  identifiers leak into "anonymised" extracts. Requiring an action per
  column costs a few lines of JSON and makes the leak impossible by default.
* **Keyed HMAC, not hashing or random ids.** Tokens stay stable across runs
  and tables (incremental refreshes join to the previous extract), and
  rotating the key unlinks every past release.
* **Release regions, not bounding boxes.** The ablation shows the textbook
  min/max box leaks a real birth date in every class. Midpoint cut boundaries
  plus padded outer bounds remove that for a small, measured utility cost.
* **Verify from the written text.** The k / l check runs on the CSV parsed
  back from disk, not the in-memory partitioning, so a formatting bug (e.g.
  two classes printing the same range) can't silently weaken the release.
* **Prosecutor attack as the privacy metric.** It's the strongest of the
  standard linkage models: the attacker knows the target is in the extract.
  It's scored against ground truth that only the evaluator holds
  (`_truth_patient_id` and the key).
* **Rows sorted by output values.** Input order (often insertion or signup
  order) is itself a quasi-identifier.

## Limitations

* **Only the subject table's QIs are anonymised.** Child tables can still
  act as quasi-identifiers: the number of encounters per patient, rare cost
  values, and visit dates known to an attacker within ±180 days. A real
  release should treat per-patient aggregates as QIs or generalise them too.
* **Date shifting breaks cross-patient seasonality** (a flu season smears
  over ±6 months). It's a trade-off, not a free lunch.
* **The protection is syntactic, not formal.** k-anonymity and distinct
  l-diversity don't stop skewness or similarity attacks (e.g. three distinct
  but all serious diagnoses), composition across several releases, or
  attackers with knowledge beyond the declared QIs. For formal guarantees on
  aggregates, use differential privacy instead. This project doesn't
  implement it.
* **Region cells still have a small residual exposure** (0.039) when two
  members' birthdays are adjacent days and no midpoint exists. Zip and
  numeric ranges aren't checked for this at all.
* **Single-machine, in-memory.** Mondrian is O(n log n) in practice, and
  5,000 subjects release in about 0.45 s. Tens of millions of rows would need
  a partitioned or SQL implementation.
* **The PII scan is heuristic.** It catches patterns and values already known
  from identifier columns, not, say, a stranger's name inside a note.
  Dropping free text remains the default advice.
* **The results are on synthetic data** with a simple age/sex-dependent
  diagnosis model. Absolute numbers will differ on real populations; the
  method comparisons are what the benchmark supports.
