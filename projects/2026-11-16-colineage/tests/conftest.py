import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colineage.graph import LineageGraph  # noqa: E402
from colineage.project import Project, analyze_project  # noqa: E402

SAMPLES = ROOT / "samples"


def load(name: str):
    project = Project.load(SAMPLES / name)
    results = analyze_project(project)
    return project, results, LineageGraph(results, set(project.sources))


@pytest.fixture(scope="session")
def shop():
    return load("shop")


@pytest.fixture(scope="session")
def saas():
    return load("saas")
