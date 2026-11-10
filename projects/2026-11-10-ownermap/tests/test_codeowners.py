from ownermap.codeowners import CodeOwners, owner_kind


def owners(text, path):
    return CodeOwners.parse(text).owners_of(path)


# Each test reproduces a statement from GitHub's CODEOWNERS documentation.
def test_default_owner_and_last_match_wins():
    co = "*  @global\n*.js @js-owner #This is an inline comment.\n"
    assert owners(co, "README.md") == ("@global",)
    assert owners(co, "src/deep/app.js") == ("@js-owner",)


def test_email_owner():
    assert owners("*.go docs@example.com\n", "cmd/main.go") == ("docs@example.com",)


def test_anchored_directory_owns_subtree():
    co = "/build/logs/ @doctocat\n"
    assert owners(co, "build/logs/a.txt") == ("@doctocat",)
    assert owners(co, "build/logs/deep/b.txt") == ("@doctocat",)
    assert owners(co, "x/build/logs/a.txt") == ()


def test_docs_star_is_one_level_only():
    co = "docs/* docs@example.com\n"
    assert owners(co, "docs/getting-started.md") == ("docs@example.com",)
    assert owners(co, "docs/build-app/troubleshooting.md") == ()


def test_unanchored_directory_anywhere():
    co = "apps/ @octocat\n"
    assert owners(co, "apps/x.py") == ("@octocat",)
    assert owners(co, "src/apps/y/z.py") == ("@octocat",)
    assert owners(co, "apps") == ()  # a *file* named apps is not a directory


def test_doublestar_logs():
    co = "**/logs @octocat\n"
    for p in ["build/logs/a", "scripts/logs/b", "deeply/nested/logs/c", "logs/d"]:
        assert owners(co, p) == ("@octocat",)


def test_explicitly_unowned_subdirectory():
    co = "/apps/ @octocat\n/apps/github\n"
    assert owners(co, "apps/web/x") == ("@octocat",)
    assert owners(co, "apps/github/x") == ()
    rule = CodeOwners.parse(co).deciding_rule("apps/github/x")
    assert rule is not None and rule.line == 2


def test_override_subdirectory():
    co = "/apps/ @octocat\n/apps/github @doctocat\n"
    assert owners(co, "apps/github/x") == ("@doctocat",)


def test_unsupported_syntax_and_invalid_owner_lines_are_skipped():
    co = CodeOwners.parse("* @a\n!/vendor/ @b\n/s/[ab].sh @c\n\\#x @d\n/t/ @bad!\n/u/ @e\n")
    codes = [(p.line, p.code) for p in co.problems]
    assert codes == [(2, "unsupported-syntax"), (3, "unsupported-syntax"), (4, "unsupported-syntax"),
                     (5, "invalid-owner")]
    assert [r.line for r in co.rules] == [1, 6]
    assert co.owners_of("vendor/x") == ("@a",)


def test_escaped_space_in_path():
    co = CodeOwners.parse("/My\\ Docs/ @w @x\n")
    assert co.rules[0].owners == ("@w", "@x")
    assert co.owners_of("My Docs/a.txt") == ("@w", "@x")


def test_owner_kinds():
    assert owner_kind("@octocat") == "user"
    assert owner_kind("@octo-org/octocats") == "team"
    assert owner_kind("docs@example.com") == "email"
    assert owner_kind("@-bad") is None and owner_kind("octocat") is None
