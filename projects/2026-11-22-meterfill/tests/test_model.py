import numpy as np
import pytest

from meterfill.allocate import coverage
from meterfill.clean import clean_meter
from meterfill.evaluate import group_rows, split_meters
from meterfill.features import FEATURES, feature_matrix
from meterfill.model import ShapeModel, choose_windows, drop_windows, hours_inside, predict_hourly, train_model


@pytest.fixture(scope="module")
def rows(small_world):
    return group_rows(small_world.readings)


def test_split_is_disjoint_and_deterministic(small_world):
    a = split_meters(small_world.meter_ids, 3)
    b = split_meters(small_world.meter_ids, 3)
    assert a == b
    train, test = a
    assert not set(train) & set(test)
    assert len(test) == round(0.3 * len(small_world.meter_ids))


def test_windows_are_fully_actual_and_separated(small_world, rows):
    w = small_world
    cov = coverage(clean_meter(rows[w.meter_ids[0]]), w.h0, w.nh)
    wins = choose_windows(cov, np.random.default_rng(1), 5)
    assert wins
    for a, b in wins:
        i0, i1 = (a - w.h0) // 3600, (b - 1 - w.h0) // 3600
        assert cov.full_actual[i0 : i1 + 1].all()
    for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
        assert a2 >= b1 + 6 * 3600


def test_features_do_not_see_masked_hours(small_world, rows):
    """Leakage check: what happened inside a masked window must not change the features."""
    w = small_world
    r = rows[w.meter_ids[1]]
    cov = coverage(clean_meter(r), w.h0, w.nh)
    wins = choose_windows(cov, np.random.default_rng(2), 3)
    masked = drop_windows(r, wins)
    X1, _ = feature_matrix(coverage(clean_meter(masked), w.h0, w.nh), w.temps)
    # different (but still monotone, same endpoints) consumption inside the windows
    a, b = wins[0]
    inside = [(t, v) for t, v in r if a <= t < b]
    tweaked = sorted(masked + [(t, inside[0][1]) for t, _ in inside])
    X2, _ = feature_matrix(coverage(clean_meter(drop_windows(tweaked, wins)), w.h0, w.nh), w.temps)
    assert X1.shape == (w.nh, len(FEATURES))
    assert np.array_equal(X1, X2)
    assert hours_inside(wins, w.h0, w.nh).size > 0


def test_train_predict_save_load(tmp_path, small_world, rows):
    w = small_world
    train, test = split_meters(w.meter_ids, 0)
    m = train_model({k: rows[k] for k in train}, w.temps, w.h0, w.nh, seed=0, rounds=1)
    assert m.meta["train_rows"] > 500
    cov = coverage(clean_meter(rows[test[0]]), w.h0, w.nh)
    pred = predict_hourly("model", cov, w.temps, m)
    assert pred.shape == (w.nh,) and (pred > 0).all()
    path = tmp_path / "m.joblib"
    m.save(path)
    m2 = ShapeModel.load(path)
    assert np.allclose(predict_hourly("model", cov, w.temps, m2), pred)


def test_model_method_requires_model(small_world, rows):
    w = small_world
    cov = coverage(clean_meter(rows[w.meter_ids[0]]), w.h0, w.nh)
    with pytest.raises(ValueError):
        predict_hourly("model", cov, w.temps, None)
    assert np.allclose(predict_hourly("uniform", cov, w.temps), cov.mean_rate())
