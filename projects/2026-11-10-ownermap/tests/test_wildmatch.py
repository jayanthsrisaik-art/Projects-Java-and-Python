import pytest

from ownermap.wildmatch import translate, wildmatch

# (pattern, text, expected) with WM_PATHNAME, mostly mirroring git's t3070-wildmatch.
CASES = [
    ("foo", "foo", True),
    ("foo", "bar", False),
    ("???", "foo", True),
    ("*", "foo", True),
    ("f*", "foo", True),
    ("*f", "foo", False),
    ("*foo*", "foo", True),
    ("*", "foo/bar", False),          # single star never crosses '/'
    ("foo/*", "foo/bar", True),
    ("foo/*", "foo/bar/baz", False),
    ("foo?bar", "foo/bar", False),
    ("**/foo", "foo", True),          # zero leading directories
    ("**/foo", "a/b/foo", True),
    ("foo/**", "foo/a/b", True),
    ("foo/**", "foo", False),
    ("foo/**/bar", "foo/bar", True),  # zero middle directories
    ("foo/**/bar", "foo/x/y/bar", True),
    ("foo**bar", "foo/bar", False),   # non-segment ** behaves like *
    ("foo**bar", "fooxbar", True),
    ("**foo", "a/foo", False),
    ("\\*", "*", True),
    ("\\*", "x", False),
    ("[ab]", "a", True),
    ("[!ab]", "c", True),
    ("[^ab]", "a", False),
    ("[!a]", "/", False),             # brackets never match '/' with WM_PATHNAME
    ("a[]]b", "a]b", True),
    ("a[]-]b", "a-b", True),
    ("[a-c]", "b", True),
    ("[c-a]", "b", False),            # reversed range matches nothing
    ("[a-]", "-", True),
    ("[-b]", "-", True),
    ("[[:alpha:]]", "Q", True),
    ("[[:digit:]x]", "x", True),
    ("[[:digit:]x]", "a", False),
    ("[[:]", "[", True),              # "[:" without ":]" is a plain '['
    ("[[:]", ":", True),
    ("[\\]]", "]", True),
    ("x\\ y", "x y", True),
]


@pytest.mark.parametrize("pattern,text,expected", CASES)
def test_wildmatch_cases(pattern, text, expected):
    assert wildmatch(pattern, text) is expected


@pytest.mark.parametrize("pattern", ["[ab", "a[", "[[:bogus:]]", "foo\\", "[\\"])
def test_patterns_git_can_never_match(pattern):
    t = translate(pattern)
    assert t.regex is None and t.reason
    assert not wildmatch(pattern, "a") and not wildmatch(pattern, pattern)


def test_matching_is_bytewise_like_git():
    # 'é' is two bytes in UTF-8, so a single '?' cannot match it.
    assert not wildmatch("?", "é")
    assert wildmatch("??", "é")


def test_basename_mode_lets_star_cross_slash():
    assert wildmatch("a*b", "a/x/b", pathname=False)
    assert not wildmatch("a*b", "a/x/b", pathname=True)


def test_literal_prefix_quirk():
    # git's match_pathname strips "a" before calling wildmatch, so ** is then segment-initial.
    assert translate("a**/**", segment_start=1).regex != translate("a**/**").regex
