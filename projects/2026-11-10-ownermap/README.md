# ownermap — CODEOWNERS you can trust

`ownermap` answers three questions teams ask about a `CODEOWNERS` file, and gets the
path-matching underneath them exactly right:

1. **Who owns this path, and which line decided?** (`who`)
2. **What is broken, dead or silently overridden in this file?** (`lint`, `coverage`)
3. **Who is the smallest set of people I must ask to review this PR?** (`review`)

All three rest on a port of git's `wildmatch` + `.gitignore` semantics that is
**differentially tested against real `git check-ignore`**: 0 mismatches on 13,121
randomly generated queries, where a typical `fnmatch`-based matcher is wrong on 10.85%.

Python standard library only. No network, no GitHub API.

## The problem

CODEOWNERS files rot quietly. Nobody notices that:

* a rule was re-declared further down and now **overrides a more specific rule** in between
  (in `samples/CODEOWNERS`, re-declaring `/services/api/` on line 13 silently takes billing
  away from `@acme/payments` on line 11);
* `/Docs/legal/` never matches because the directory is `docs/` (paths are case-sensitive);
* `!/vendor/` or `/scripts/[ab]*.sh` use gitignore syntax GitHub does not support, so the
  line is skipped;
* a team listed as owner has no members, so its files can never be approved;
* a pattern like `docs/*` owns `docs/a.md` but **not** `docs/guides/b.md`.

And when a PR touches ten files with different owners, "request everyone listed" is
wasteful; the minimum set of reviewers is a hitting-set problem that greedy picking gets
wrong surprisingly often.

## Architecture

```
ownermap/
  wildmatch.py    git wildmatch -> byte regex (brackets, POSIX classes, **, escapes, aborts)
  pathpattern.py  .gitignore line parsing + evaluator (negation, dir-only, anchoring,
                  parent-directory exclusion, git's literal-prefix quirk)
  gitoracle.py    differential harness: random tree + random .gitignore -> git check-ignore
  codeowners.py   CODEOWNERS parser, owner validation, last-match-wins resolution,
                  directory-owns-subtree rule, whole-tree Resolution with caching
  lint.py         syntax / registry / tree-aware checks, coverage summary
  registry.py     local JSON: users, team membership, e-mail aliases
  review.py       requirement building, exact branch-and-bound hitting set, greedy baseline,
                  brute-force oracle
  bench.py        measured benchmarks (git differential; exact vs greedy reviewers)
  cli.py          `python -m ownermap who|lint|coverage|review|check-ignore|bench`
```

Data flow for `review`: changed path → deciding rule (last match) → owners → approvers
(team members / e-mail alias, author removed) → requirement sets → superset reduction →
branch and bound (pivot on the smallest open requirement, disjoint-packing lower bound,
greedy upper bound) → reviewers + which files each one covers.

## Setup

```bash
cd projects/2026-11-10-ownermap
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # only pytest; the tool itself is stdlib-only
```

`git` on `PATH` is optional; it enables the differential tests and `bench ignore`.

## Usage

```bash
# Lint against a file list (or --git REPO for `git ls-files`, or --root DIR to walk)
python -m ownermap lint -c samples/CODEOWNERS --files samples/files.txt --registry samples/registry.json
python -m ownermap lint ... --format json --strict     # exit 1 on errors (and on warnings with --strict)

# Who owns what, and why
python -m ownermap who -c samples/CODEOWNERS --why docs/index.md services/worker/Dockerfile

# Coverage per top-level directory (CI gate with --min-coverage)
python -m ownermap coverage -c samples/CODEOWNERS --files samples/files.txt --min-coverage 90

# Minimum reviewer set for a change; the author's approval does not count
python -m ownermap review -c samples/CODEOWNERS --changed samples/changed.txt \
    --registry samples/registry.json --author erin

# git check-ignore for a root .gitignore (directories end with "/")
python -m ownermap check-ignore --gitignore samples/sample.gitignore -v out/ logs/important.txt

# Benchmarks (results are written to results/)
python -m ownermap bench ignore --cases 400 --seed 7 --out results/ignore_differential.json
python -m ownermap bench review --cases 500 --seed 11 --out results/review_benchmark.json
```

### Sample output

`samples/expected/` holds the exact output of each command on the sample inputs
(regenerate with `python samples/regenerate_expected.py`; `tests/test_cli.py` checks them).
`lint` on the sample:

```
samples/CODEOWNERS:10: warning: [shadowed] '/services/api/' matches 4 file(s) but later rules (line 13, 22) override all of them
samples/CODEOWNERS:11: warning: [shadowed] '/services/api/billing/' matches 2 file(s) but later rules (line 13) override all of them
samples/CODEOWNERS:13: warning: [duplicate-pattern] pattern '/services/api/' already appears on line 10; the earlier rule can never decide ownership
samples/CODEOWNERS:17: warning: [no-match] '/Docs/legal/' matches no file in the tree (it would match if case were ignored: CODEOWNERS paths are case-sensitive)
samples/CODEOWNERS:25: info: [no-owners] '/web/src/generated/' has no owners: matching files are explicitly unowned
samples/CODEOWNERS:28: error: [unsupported-syntax] negation ('!/vendor/') is not supported in CODEOWNERS; line skipped
samples/CODEOWNERS:29: error: [unsupported-syntax] character ranges ('/scripts/[ab]*.sh') are not supported in CODEOWNERS; line skipped
samples/CODEOWNERS:30: error: [invalid-owner] '@not_a_valid-handle!' is not a @user, @org/team or e-mail; line skipped
samples/CODEOWNERS:31: warning: [empty-team] team '@acme/legacy' has no members, nobody can approve for it
samples/CODEOWNERS:31: warning: [no-match] '/legacy/' matches no file in the tree
3 error(s), 6 warning(s), coverage 95.7% (22/23 files owned)
```

`review` for 9 changed files, author `@erin`: 5 reviewers (`@dana @frank @gail @heidi @paul`),
with the files each one covers, and `web/src/generated/api.ts` reported as `no-owners`.

## Measured results

All numbers below were produced by the commands above in this environment (Python 3.11.18,
git 2.43.0); the JSON files are in `results/`.

**.gitignore evaluator vs `git check-ignore --no-index -v`** (`results/ignore_differential.json`):
400 random cases (random trees with spaces, `[`, `!`, `#`, `é` in names; 2–7 random patterns
per file mixing `**`, brackets, classes, escapes, negation, anchoring, dir-only, trailing spaces),
13,121 file and directory queries:

| evaluator | wrong verdicts | wrong deciding line |
|---|---|---|
| ownermap | **0** | **0** |
| naive fnmatch (last match wins, basename if no `/`, dir-only, negation) | 1,423 (10.85%), in 248/400 cases | – |

The differential run found one real discrepancy during development: git's `match_pathname`
compares the literal prefix with `strncmp` and passes only the remainder to wildmatch, so in
`a**/**` the `**` becomes segment-initial and the pattern matches the directory `a`. This
quirk is now reproduced and covered by tests.

**Reviewer planning, exact vs greedy** (`results/review_benchmark.json`): 500 synthetic PRs
(8–30 people, 3–10 overlapping teams, 5–40 changed files, 40% of files with two owners):

| | mean reviewers | instances worse than optimal | mean time |
|---|---|---|---|
| exact branch and bound | 8.96 | – | 0.081 ms |
| greedy (most-covering first) | 9.138 | 83 (16.6%), 89 extra reviewers in total | 0.028 ms |

The exact search never hit its node budget (max 91 nodes) and matched the brute-force oracle
on all 146 instances small enough to enumerate. These are synthetic instances; real repos
with fewer, larger teams will usually show a smaller gap.

## Tests

```bash
python -m pytest -q     # 135 tests
```

* `test_wildmatch.py` — git t3070-style cases: `**` placement, bracket parsing edge cases
  (`[]]`, `[[:]`, reversed ranges), unmatchable patterns, byte-wise `?`.
* `test_gitignore.py` — parsing rules, parent-exclusion, plus differential tests against
  real git (23 hand-picked quirky patterns and 80 random cases; skipped without git).
* `test_codeowners.py` — every behaviour from GitHub's CODEOWNERS example file, inline
  comments, invalid/unsupported lines, escaped spaces.
* `test_lint.py`, `test_review.py` — each lint check; exact solver equals the brute-force
  oracle (size *and* tie-break) on 25 seeds; greedy's failure mode; budget reporting;
  author exclusion.
* `test_cli.py` — golden outputs in `samples/expected/` and exit codes.

## Design decisions

* **Byte regexes compiled from a port, not `fnmatch`.** Git matches bytes, and its bracket
  parser has behaviour (`[[:]`, reversed ranges, aborts on `[[:bogus:]]`) no regex dialect
  shares, so the bracket loop is ported statement by statement and emitted as explicit
  `\xHH` byte classes.
* **Correctness is established against git, not against my reading of the docs.** The random
  generator deliberately mixes the awkward syntax, and the line number of the deciding
  pattern is compared, not only the verdict.
* **CODEOWNERS ≠ .gitignore.** GitHub does not support `!`, `[ ]` or `\#` in CODEOWNERS;
  `ownermap` reports such lines as errors and skips them instead of guessing. Inline `#`
  comments are supported, as in GitHub's own example file.
* **Directory rules own the subtree, except patterns ending in `*`.** This is the rule that
  reproduces every example in GitHub's documentation (`apps/`, `/build/logs/`, `**/logs`,
  `/apps/github` vs `docs/*`).
* **"Shadowed" is defined on the actual tree**: a rule that matches files but decides none.
  That catches the dangerous case (a re-declared broad rule overriding a specific one) that
  a pure duplicate check misses.
* **Exact over heuristic for reviewers**, because instances are tiny: supersets are removed
  first, the smallest open requirement is the branching pivot and a disjoint-packing bound
  prunes. A node budget guarantees termination; if hit, the result is flagged
  `optimal: false` instead of claimed optimal. Ties are broken lexicographically so output
  is deterministic.
* **Without a registry, teams are treated as single reviewers**, which is what "request
  review from team X" means on GitHub.

## Limitations

* The .gitignore evaluator handles a **single root-level** `.gitignore`; nested
  `.gitignore` files, `.git/info/exclude`, `core.excludesFile` and `core.ignorecase=true`
  are not modelled.
* GitHub's CODEOWNERS matcher is closed source. The semantics here are validated against
  GitHub's documented examples, not against GitHub itself; edge cases outside those
  examples (e.g. `docs/*.md` matching a directory named `x.md`) may differ.
* Only one CODEOWNERS file is read; GitHub's lookup order (`.github/`, root, `docs/`) and
  the 3 MB size limit are not checked.
* The registry is a local JSON file; team membership and write access are not fetched from
  GitHub, and nested teams are not expanded.
* `review` models "one approval from any code owner per file"; branch-protection settings
  such as required approval counts or dismissing stale reviews are out of scope.
* The reviewer benchmark uses synthetic organisations, so the greedy gap is indicative only.
