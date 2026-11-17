import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlcritic.db import build_db, connect_readonly  # noqa: E402


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return build_db(tmp_path_factory.mktemp("wh") / "warehouse.db")


@pytest.fixture(scope="session")
def con(db_path):
    return connect_readonly(db_path)
