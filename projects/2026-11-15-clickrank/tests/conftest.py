import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clickrank.catalog import build_catalog, vocabulary  # noqa: E402
from clickrank.features import SearchIndex  # noqa: E402
from clickrank.ingest import ingest  # noqa: E402
from clickrank.simulate import simulate  # noqa: E402


class World:
    """A small simulated world shared by the slower tests (built once per session)."""

    def __init__(self, tmp: Path):
        self.catalog = build_catalog(seed=7)
        self.index = SearchIndex(self.catalog.products, vocabulary())
        self.lines, self.truth = simulate(self.catalog, self.index, 25000, swap_rate=0.3, seed=7)
        self.db = tmp / "world.db"
        self.report = ingest(self.lines, self.db)

    def conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db)


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> World:
    return World(tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="session")
def samples() -> Path:
    return ROOT / "samples"
