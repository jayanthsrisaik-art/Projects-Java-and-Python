"""Regenerate samples/expected/* from the current implementation (review diffs before committing)."""

import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ownermap.cli import main  # noqa: E402

# name -> argv (paths relative to the project root)
CASES = {
    "lint.txt": ["lint", "-c", "samples/CODEOWNERS", "--files", "samples/files.txt", "--registry", "samples/registry.json"],
    "lint.json": ["lint", "-c", "samples/CODEOWNERS", "--files", "samples/files.txt", "--registry", "samples/registry.json", "--format", "json"],
    "coverage.txt": ["coverage", "-c", "samples/CODEOWNERS", "--files", "samples/files.txt"],
    "who.txt": ["who", "-c", "samples/CODEOWNERS", "--why", "docs/index.md", "docs/guides/setup.md",
                "services/worker/Dockerfile", "web/src/generated/api.ts", "vendor/lib/x.js"],
    "review.txt": ["review", "-c", "samples/CODEOWNERS", "--changed", "samples/changed.txt",
                   "--registry", "samples/registry.json", "--author", "erin"],
    "review.json": ["review", "-c", "samples/CODEOWNERS", "--changed", "samples/changed.txt",
                    "--registry", "samples/registry.json", "--author", "erin", "--format", "json"],
    "check_ignore.txt": ["check-ignore", "--gitignore", "samples/sample.gitignore", "-v", "a/build/", "app.log",
                         "keep.log", "dist/", "dist/x/y", "logs/important.txt", "out/", "Temp/", "#notes.md", "src/main.py"],
}


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    for name, argv in CASES.items():
        code, out = run(argv)
        with open(os.path.join(HERE, "expected", name), "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"{name}: exit {code}")
