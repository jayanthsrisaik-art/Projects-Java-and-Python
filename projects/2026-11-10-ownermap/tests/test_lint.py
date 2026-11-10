import os

from ownermap.codeowners import CodeOwners
from ownermap.lint import lint
from ownermap.registry import Registry

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def codes(report):
    return [(p.line, p.code) for p in report.problems]


def test_shadowed_and_no_match():
    co = CodeOwners.parse("* @a\n/src/ @b\n/src/ @c\n/nope/ @d\n")
    r = lint(co, ["src/x.py", "README"])
    assert (2, "shadowed") in codes(r) and (3, "duplicate-pattern") in codes(r)
    assert (4, "no-match") in codes(r)
    assert (1, "shadowed") not in codes(r)


def test_case_mismatch_hint():
    co = CodeOwners.parse("/Docs/ @a\n")
    [p] = [p for p in lint(co, ["docs/a.md"]).problems if p.code == "no-match"]
    assert "case" in p.message


def test_registry_checks():
    reg = Registry.from_dict({"teams": {"@o/t": []}, "users": ["@a"]})
    r = lint(CodeOwners.parse("* @a @o/t @ghost\n"), None, reg)
    assert codes(r) == [(1, "unknown-owner"), (1, "empty-team")]  # errors sort first


def test_coverage_counts():
    co = CodeOwners.parse("/a/ @x\n/a/gen/\n")
    r = lint(co, ["a/1", "a/gen/2", "b/3"])
    assert (r.total_files, r.owned_files, r.explicitly_unowned) == (3, 1, 1)
    assert r.unowned_dirs == [("a/", 1), ("b/", 1)]


def test_sample_lint_findings():
    co = CodeOwners.from_file(os.path.join(SAMPLES, "CODEOWNERS"))
    with open(os.path.join(SAMPLES, "files.txt")) as fh:
        files = fh.read().split()
    r = lint(co, files, Registry.from_file(os.path.join(SAMPLES, "registry.json")))
    assert r.errors == 3
    got = codes(r)
    # The re-declared /services/api/ (line 13) silently overrides the billing rule (line 11).
    assert (10, "shadowed") in got and (11, "shadowed") in got and (13, "duplicate-pattern") in got
    assert (17, "no-match") in got and (31, "empty-team") in got
