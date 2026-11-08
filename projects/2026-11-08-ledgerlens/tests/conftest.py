import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledgerlens.cli import train_model  # noqa: E402
from ledgerlens.synth import generate  # noqa: E402


@pytest.fixture(scope="session")
def full_rows():
    rows, _ = generate(seed=7)  # identical to `python -m ledgerlens generate` defaults
    return [t.__dict__ for t in rows]


@pytest.fixture(scope="session")
def model(full_rows):
    return train_model(full_rows)


@pytest.fixture(scope="session")
def small_rows():
    rows, _ = generate(seed=11, n_accounts=20)
    return [t.__dict__ for t in rows]
