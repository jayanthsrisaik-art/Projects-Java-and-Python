import numpy as np

from priorshift.calibration import BCTS, expected_calibration_error, nll
from priorshift.detect import detect_shift


def overconfident(P, power):
    Q = P**power
    return Q / Q.sum(axis=1, keepdims=True)


def test_bcts_undoes_temperature_distortion(world):
    P, y, _ = world.batch(world.train_priors, 20_000, seed=10)
    distorted = overconfident(P, 3.0)  # log-probs scaled by 3, i.e. T should come out ≈ 3
    cal = BCTS().fit(distorted, y)
    assert 2.6 < cal.temperature < 3.4
    fixed = cal.transform(distorted)
    assert nll(fixed, y) < nll(distorted, y)
    assert expected_calibration_error(fixed, y) < expected_calibration_error(distorted, y)
    np.testing.assert_allclose(fixed.sum(axis=1), 1.0)


def test_bcts_learns_class_bias(world):
    P, y, _ = world.batch(world.train_priors, 20_000, seed=11)
    wrong = P * np.array([1.0, 1.0, 4.0])  # model thinks class 2 is 4x more common than it is
    wrong /= wrong.sum(axis=1, keepdims=True)
    cal = BCTS().fit(wrong, y)
    assert cal.bias[2] < cal.bias[0] - 1.0
    assert nll(cal.transform(wrong), y) < nll(wrong, y)


def test_shift_test_false_alarm_rate_under_null(world):
    alarms = []
    for s in range(200):
        P, _, _ = world.batch(world.train_priors, 400, seed=100 + s)
        alarms.append(detect_shift(P, world.train_priors, alpha=0.05, min_tv=0.0).p_value < 0.05)
    rate = np.mean(alarms)
    assert 0.01 <= rate <= 0.11  # nominal 5%


def test_shift_test_fires_on_real_shift_and_min_tv_gate(world):
    P, _, _ = world.batch([0.2, 0.3, 0.5], 1_000, seed=12)
    t = detect_shift(P, world.train_priors, alpha=0.01, min_tv=0.05)
    assert t.shifted and t.p_value < 1e-6 and t.tv_distance > 0.2
    t2 = detect_shift(P, world.train_priors, alpha=0.01, min_tv=0.9)
    assert not t2.shifted  # significant but below the effect-size gate
