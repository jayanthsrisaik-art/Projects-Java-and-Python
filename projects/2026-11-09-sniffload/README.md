# sniffload: load messy CSV exports without guessing the locale for every column

CSV files from banks, ERPs and spreadsheets come in many forms: `;` or tab delimiters, cp1252 or
UTF-16 encodings, `1.234,56` vs `1,234.56`, `03/04/2024` meaning March or April, `ja`/`nein`
booleans, `-` or `N/A` for nulls, ZIP codes with leading zeros, and the odd `#REF!`. The usual
loaders either assume one locale for everything or keep every column as text.

sniffload works out each file's encoding, delimiter, header, null tokens and column types,
**including the decimal separator and the day/month order**. It loads the result into SQLite
with typed columns, and rows that don't fit go to a rejects table with a line number and a reason.

The main idea: **a single column often can't decide its own format, but the file can.**
`Menge: 1.200` could mean 1200 or 1.2. When another column in the same file contains `1.234,56`,
`,` must be the decimal separator, so `1.200` is 1200. In the same way, one `31.12.2024` in any
date column fixes the day/month order for every ambiguous date column in that file. Each
decision is recorded with how it was made (`data`, `file`, `delimiter-prior`, `locale-prior`,
`default`), and ambiguous ones produce a warning.

```
$ python -m sniffload infer samples/de_orders.csv
encoding : cp1252 (not valid UTF-8; decodes as cp1252)
delimiter: ;   header: True
decimal  : ',' (file)   date order: DMY (file)
columns  :
  kunde                string    nulls=0   <- 'Kunde'
  betrag               decimal   dec=',' scale=2 by=data nulls=1   <- 'Betrag'
  menge                integer   dec=',' by=file nulls=0 bad=1   <- 'Menge'
  bestelldatum         date      format=%d.%m.%Y by=data nulls=0   <- 'Bestelldatum'
  bezahlt              boolean   nulls=0   <- 'Bezahlt'
  notiz                string    nulls=10   <- 'Notiz'
warning  : 1 record(s) do not have 6 fields
warning  : menge: decimal separator ambiguous in this column (e.g. '1.000'); used ',' (file)
```

Here `bestelldatum` decides DMY by itself because one row says `31.12.2024`. Every other date in
that file has day ≤ 12.

## Architecture

```
bytes ──► encoding.py ──► infer.py ───────────────────────────────► load.py ─────► SQLite
          BOM / UTF-16     detect_delimiter: score , ; TAB |          convert every    <table>
          strict UTF-8     (consistency, typed share, leakiness)      cell; reject      <table>__rejects
          cp1252 / latin-1 header vote ─┐                             ragged/bad rows   clean CSV
                                        ▼                             (or NULL them)    JSON report
                           columns.analyze_column (per column)
                             bool → number('.'/',') → date → datetime → string
                             marks decisive vs ambiguous
                                        │
                           _resolve_file_locale  (votes of decisive columns,
                                        │         then ';' prior, then defaults)
                                        ▼
                           columns.finalize_column → ColumnSchema (+ resolved_by, warnings)
```

| module | responsibility |
|---|---|
| `encoding.py` | BOMs, BOM-less UTF-16 via NUL positions, strict UTF-8, cp1252, latin-1 fallback |
| `values.py` | scalar parsers: null tokens, booleans (en/de/fr), numbers with `.`/`,` decimals, thousands groups (`,` `.` space NBSP `'`), currency, accounting negatives, 16 date and 7 datetime formats |
| `columns.py` | per-column analysis with a tolerance (`min_ratio`, default 0.95), leading-zero identifiers, final `ColumnSchema`, cell conversion |
| `infer.py` | delimiter scoring, header vote, name normalisation, file-level decimal/date-order resolution |
| `load.py` | full-file conversion, rejects, `on_bad=reject\|null`, locked-schema drift check, transactional SQLite write, clean CSV |
| `synth.py` / `bench.py` | ground-truth generator and benchmark against `csv.Sniffer` and pandas |
| `cli.py` | `infer`, `load`, `bench`, `synth` |

## Setup

```bash
cd projects/2026-11-09-sniffload
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pytest; pandas only needed for the pandas baselines
python -m pytest -q
```

The runtime uses only the standard library (tested on Python 3.11.18; pandas 3.0.6 for the baseline).

## Usage

```bash
# Inspect what was inferred (add --json for the full schema, -o to save it)
python -m sniffload infer samples/us_payments.csv

# Load into SQLite: table + table__rejects, plus an optional clean UTF-8 CSV and JSON report
python -m sniffload load samples/de_orders.csv --db out.db --table orders \
    --clean-csv orders.clean.csv --report orders.report.json

# Recurring feed: lock the schema once, then fail fast on header drift (exit code 2)
python -m sniffload infer samples/de_orders.csv -o orders.schema.json
python -m sniffload load samples/de_orders_v2.csv --db out.db --schema orders.schema.json
# schema drift: new columns: ['Kanal']

# Quality gate: write nothing and exit 3 if more than 5% of rows are rejected
python -m sniffload load samples/de_orders.csv --db out.db --max-reject-ratio 0.05

# Keep rows with bad cells, setting those cells to NULL
python -m sniffload load samples/de_orders.csv --db out.db --on-bad null

# Benchmark, and write synthetic files to look at
python -m sniffload bench --files 300 --seed 0 --out results.json
python -m sniffload synth --out /tmp/messy --n 10
```

Fallbacks for when nothing in the file decides: `--date-order {MDY,DMY,YMD}` (default MDY) and
`--decimal {.,,}` (default `.`). `--min-ratio` sets how many non-null values must parse for a
column to get a type.

### Samples and expected outputs

| input | what it exercises | expected outputs |
|---|---|---|
| `samples/de_orders.csv` | cp1252, `;`, CRLF, `1.234,56`, ambiguous `1.200` quantities, DMY settled by one row, `ja`/`nein`, `NA`/`-` nulls, quoted `;`, a `#REF!` cell, a ragged row | `samples/expected/de_orders.*` |
| `samples/us_payments.csv` | UTF-8 BOM, `"$1,250.00"`, `(45.10)`, M/D/YYYY without padding, `YES`/`No`, ZIPs with leading zeros, a multi-line quoted memo | `samples/expected/us_payments.*` |
| `samples/readings_noheader.psv` | `\|` delimiter, no header, ISO datetimes with `Z`, offsets and a space separator | `samples/expected/readings_noheader.*` |
| `samples/ambiguous_dates.csv` | every date has day ≤ 12 and nothing else decides → MDY default with a warning | `samples/expected/ambiguous_dates.*` |
| `samples/de_orders_v2.csv` | header gained a column → `SchemaDriftError` / exit 2 with a locked schema | (tested in `tests/test_load.py`) |

Each input has three expected files: `*.schema.txt` (output of `infer`), `*.clean.csv`
(normalised values: ISO dates, `.` decimals, `true`/`false`) and `*.rejects.json` (line, raw
record, reasons). `tests/test_load.py` compares against them byte for byte.
`python samples/regenerate_expected.py` rebuilds them.

## Evaluation

`sniffload.synth` generates CSV files together with the exact correct answer for each one.
Locales are US, UK, DE, FR (NBSP thousands) and ISO. Delimiters are `,` `;` TAB and `|`, matched
to locale the way spreadsheets export them. Encodings are UTF-8, UTF-8-BOM, cp1252 and UTF-16.
Line endings are LF or CRLF, and 15% of files have no header. Values include 5% nulls written
with five different tokens, quoted fields containing delimiters, quotes, tabs and newlines,
and ZIP codes. 15% of files are small (6–12 rows). Half of the larger files get 1–2 corrupt
cells (`#REF!`, `12..5`, `TBD`, …) in typed columns.

Metrics: delimiter, header and decode accuracy per file. Column type accuracy covers 6 logical
types. Good-cell accuracy counts a non-corrupt cell as correct only if its exact value is
recovered; cells lost to a wrongly rejected row count as wrong. Bad-cell catch rate is the share
of corrupt cells that were rejected or nulled instead of loaded as a value.

**Measured** with `python -m sniffload bench --files 300 --seed 0` (full output in
`results/benchmark_300files_seed0.json`):

| system | delimiter | header | decode | column type | good cells | bad-cell catch | good rows rejected |
|---|---|---|---|---|---|---|---|
| **sniffload** | 1.000 | 1.000 | 1.000 | **0.994** | **0.995** | 0.979 | 0 |
| sniffload, no file-level resolution (ablation) | 1.000 | 1.000 | 1.000 | 0.979 | 0.990 | 0.979 | 0 |
| stdlib `csv.Sniffer` | 0.983 | 0.917 | (given) | – | – | – | – |
| pandas 3.0 `read_csv`, given true encoding/delimiter/header | (given) | (given) | (given) | 0.512 | 0.530 | 0.000 | – |
| same + true `decimal=`/`thousands=` | (given) | (given) | (given) | 0.565 | 0.552 | 0.000 | – |

File-level resolution matters most for the DE subset, where `.` is the thousands separator
(55 files): column type accuracy 0.983 vs 0.901 for the ablation, and good cells 0.992 vs 0.967.
For US/UK/ISO/FR files the two variants score the same, because `,`-thousands values and the
`.` default happen to agree, and FR's NBSP grouping is never ambiguous.

Why the pandas baselines score low, even when given the right dialect and locale (column types
correct per kind, measured on the same 300 files): dates 0/152 and datetimes 0/146, since pandas
only parses dates when told which columns; booleans 3/143 (`yes`/`no`, `Y`/`N` stay text); ZIP
codes 95/143 (leading zeros dropped as int); amounts 48/149 and quantities 49/165. A single `-`
null token or one `#REF!` makes pandas read the whole column as text. pandas never rejects a row,
so its bad-cell catch rate is 0. This comparison measures pandas *defaults*. With per-column
`dtype=`, `parse_dates=`, `dayfirst=`, `true_values=` and `na_values=`, pandas could do much
better, but writing those arguments for every file is exactly the work sniffload automates.

Where sniffload's remaining errors come from (every miss on the 300 files, found by inspection):
* **No evidence anywhere in the file.** 4 DE files use `,` or TAB delimiters (so the `;` prior
  doesn't apply) and contain no decisive numeric column. Their `1.200`-style quantities fall back
  to the `.` default and load as 1.2. This causes most of the lost good cells.
  `--decimal ,` or a locked schema fixes it.
* **ZIP codes in small files.** In 5 files of 6–10 rows, none of the ZIPs happens to have a
  leading zero, so the column is typed `integer`. Without a zero there is nothing in the data to
  go on.
* **Tolerance at the boundary.** Two columns (38 and 42 rows) each got 2 corrupt cells, which
  puts them just under 95% and makes them `string`. Their corrupt cells are then loaded as text.
  These are all 4 of the missed bad-cell catches.

## Design decisions

* **Decide per file, then per column.** Locale is a property of the export, not of a column.
  Decisive columns vote. With no votes, a `;` delimiter implies `,` decimals (Excel exports `;`
  exactly in comma-decimal locales), and a comma-decimal locale implies DMY. Only after that do
  the configurable defaults apply. Each step is recorded in `resolved_by`.
* **Parsers are strict; inference is tolerant.** `1,23` is never a number with `.` decimals
  (thousands groups have 3 digits), and `0,125` is never 125 (a leading `0` group isn't a
  thousands group). A column still gets a type if ≥ 95% of non-null values parse. The rest are
  counted as `bad_count` at inference time, and their rows are rejected at load time, so one
  `#REF!` doesn't turn a numeric column into text.
* **Leading zeros mean identifier.** `02134` is kept as text because it is a key, not a quantity.
* **Delimiter scoring is a tuple:** field-count consistency first, then the share of typed
  columns, then a penalty for text columns that keep containing another candidate delimiter,
  then width. This handles `a;1,5`, where every row also has exactly one comma. On the
  benchmark, `csv.Sniffer` picked the wrong delimiter on 5 of 300 files and sniffload on none.
* **Header vote by type mismatch.** Row 0 is a header if more typed columns fail to parse it than
  accept it. For all-text files the fallback is "row 0 is unique and never repeats in its column".
* **Loads are idempotent and atomic.** Drop and recreate in one transaction. `_line` keeps lineage
  back to the physical source line, which is correct even after multi-line quoted fields.
* **Locked schemas for recurring feeds.** `infer -o` saves the full decision, including the
  encoding. `load --schema` reuses it, so a later file that happens to be ASCII-only or
  ambiguous still parses the same way, and header changes fail loudly.

## Limitations

* Only `"` is supported as the quote character. No preamble/title rows above the header, no
  multi-row headers, and no fixed-width files.
* Date month names are English only (`%b`). Two-digit years follow Python's `%y` pivot (69–99 →
  1900s). A datetime column may mix naive and offset-aware values; they are stored as ISO text.
* An all-text headerless file whose first row happens to be unique is read as having a header.
* SQLite stores decimals as `REAL`. Exact values are in the clean CSV and the Python API
  (`Decimal`). Booleans are stored as 0/1.
* Inference reads the whole file in memory (the sample is capped by `sample_rows` for typing,
  but the load is not streamed). It suits files up to a few hundred MB, not bigger.
* The benchmark is synthetic. Real exports have quirks it doesn't model, such as mixed locales
  inside one file, Excel scientific-notation IDs (`1.23E+11`) and footer/total rows. The numbers
  above apply only to this generator.
