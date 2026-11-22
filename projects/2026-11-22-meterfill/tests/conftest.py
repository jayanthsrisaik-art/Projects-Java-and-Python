import pytest

from meterfill.simulate import SimConfig, simulate

H = 3600
T0 = 1_767_571_200  # 2026-01-05T00:00:00Z


@pytest.fixture(scope="session")
def small_world():
    return simulate(SimConfig(households=24, days=21, seed=5))
