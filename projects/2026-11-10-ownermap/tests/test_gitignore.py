import pytest

from ownermap.gitoracle import git_available, git_check_ignore, git_has_prefix_quirk, run_differential
from ownermap.pathpattern import GitIgnore, parse_gitignore, trim_trailing_spaces


def test_parse_comments_blank_crlf_and_negation():
    pats = parse_gitignore(b"# c\n\n*.log\r\n!keep.log\n\\#lit\n\\!bang\n")
    assert [p.raw for p in pats] == [b"*.log", b"!keep.log", b"\\#lit", b"\\!bang"]
    assert [p.line_no for p in pats] == [3, 4, 5, 6]
    assert pats[1].negated and not pats[3].negated


@pytest.mark.parametrize("raw,trimmed", [
    (b"foo  ", b"foo"), (b"foo\\ ", b"foo\\ "), (b"foo\\  ", b"foo\\ "), (b"a b", b"a b"), (b"foo\\", b"foo\\"),
])
def test_trailing_spaces(raw, trimmed):
    assert trim_trailing_spaces(raw) == trimmed


def test_dir_only_and_anchoring():
    gi = GitIgnore("build/\n/dist\n")
    assert gi.check("a/build", is_dir=True).ignored
    assert not gi.check("a/build", is_dir=False).ignored  # a file named build
    assert gi.check("dist", is_dir=True).ignored
    assert not gi.check("a/dist", is_dir=True).ignored    # anchored to root


def test_parent_exclusion_cannot_be_undone():
    gi = GitIgnore("logs/\n!logs/keep.txt\n")
    r = gi.check("logs/keep.txt")
    assert r.ignored and r.pattern.line_no == 1


def test_reinclude_pattern_that_works():
    gi = GitIgnore("/*\n!/src/\n")
    assert gi.check("README.md").ignored
    assert not gi.check("src/main.py").ignored
    assert gi.check("src", is_dir=True).pattern.negated


def test_prefix_quirk_follows_git_version():
    # git 2.52+ (default): '**' right after a literal prefix is not segment-initial.
    assert not GitIgnore("a**/**\n").check("a", is_dir=True).ignored
    assert GitIgnore("a**/**\n").check("ab/c").ignored
    # git < 2.52 stripped the whole prefix, so 'a**/**' also matched the directory 'a'.
    assert GitIgnore("a**/**\n", legacy_prefix_quirk=True).check("a", is_dir=True).ignored
    # patterns without the quirk shape are unaffected
    for text in ("a/**\n", "**/a/**/b\n", "/a**\n"):
        for q, d in (("a", True), ("a/b", False), ("ab", False), ("x/a/y/b", False)):
            new, old = GitIgnore(text).check(q, d), GitIgnore(text, legacy_prefix_quirk=True).check(q, d)
            assert (new.ignored, new.pattern and new.pattern.line_no) == (old.ignored, old.pattern and old.pattern.line_no)


needs_git = pytest.mark.skipif(not git_available(), reason="git not installed")

QUIRKS = [
    "a**/**", "/a**/**", "a/**", "**", "**/", "*/", "a/*/", "**/a/**/b", "[!a]*/", "?/", "\\ ", "x\\ y",
    "\\#h", "#h", "!", "/", "[[:]", "[[:bogus:]]", "a\\", "[a-]", "é?", "**a/", "a/**/",
]


@needs_git
@pytest.mark.parametrize("pattern", QUIRKS)
def test_quirky_single_patterns_agree_with_git(pattern):
    files = ["a/b", "a/a/b", "x y", "#h", "é1/z", "ab/c", "b/a/q/b", "aa/x", "a- /y", "-"]
    dirs = sorted({"/".join(f.split("/")[:k]) for f in files for k in range(1, f.count("/") + 1)})
    gi_text = "*.none\n" + pattern + "\n!a/a\n"
    truth = git_check_ignore(gi_text, files, files + dirs)
    ours = GitIgnore(gi_text, legacy_prefix_quirk=git_has_prefix_quirk())
    for q in files + dirs:
        r = ours.check(q, q in dirs)
        assert (r.ignored, r.pattern.line_no if r.pattern else 0) == truth[q], (pattern, q)


@needs_git
def test_random_differential_against_git():
    stats = run_differential(cases=80, seed=2024)
    assert stats.queries > 2000
    assert stats.ours_mismatch == 0 and stats.ours_line_mismatch == 0, stats.first_mismatch
    # The naive fnmatch baseline is measurably wrong on the same inputs.
    assert stats.naive_mismatch > 0
