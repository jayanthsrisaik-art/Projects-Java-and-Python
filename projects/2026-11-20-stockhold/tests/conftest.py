import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stockhold.bench import SimClock  # noqa: E402
from stockhold.store import SequentialIds, Store  # noqa: E402


@pytest.fixture
def clock():
    return SimClock(1_000.0)


@pytest.fixture
def store(tmp_path, clock):
    s = Store(str(tmp_path / "t.db"), hold_ttl=60, clock=clock, ids=SequentialIds())
    s.upsert_product("A", "Thing", 1000, total=3, per_customer_limit=2)
    yield s
    s.close()
