import pytest

from dqwatch.detect import psi
from dqwatch.stats import chi2_isf, chi2_sf, z_tail


@pytest.mark.parametrize("x,df,p", [(3.841459, 1, 0.05), (5.991465, 2, 0.05), (9.21034, 2, 0.01),
                                    (18.307038, 10, 0.05), (0.0, 3, 1.0)])
def test_chi2_sf_known_values(x, df, p):
    assert chi2_sf(x, df) == pytest.approx(p, abs=1e-6)


def test_chi2_isf_inverts_sf():
    for df in (1, 2, 5, 30):
        for p in (0.1, 1e-3, 3e-5):
            assert chi2_sf(chi2_isf(p, df), df) == pytest.approx(p, rel=1e-6)


def test_z_tail():
    assert z_tail(1.959964) == pytest.approx(0.025, abs=1e-6)


def test_psi_basic_properties():
    a = {"x": 50, "y": 50}
    assert psi(a, {"x": 500, "y": 500}) == pytest.approx(0.0, abs=1e-12)
    assert psi({"x": 90, "y": 10}, a) > psi({"x": 60, "y": 40}, a) > 0
    assert psi({"x": 50, "y": 50, "new": 30}, a) > 0.1  # unseen category stays finite
