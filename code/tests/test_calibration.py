import numpy as np
import pytest
from eval.calibration import (adaptive_snorm, snorm_scores, fit_platt,
                              apply_platt, fit_temperature, apply_temperature)
from eval.metrics import ece


def test_snorm_standardizes_against_cohort():
    cohort = [0.0, 0.2, 0.4, 0.6, 0.8]  # mean 0.4
    z = adaptive_snorm(0.4, cohort)
    assert abs(z) < 1e-9
    assert adaptive_snorm(0.9, cohort) > 0


def test_snorm_top_k_uses_highest_cohort_scores():
    cohort = [0.0, 0.0, 0.0, 0.5, 0.7]
    z_all = adaptive_snorm(0.6, cohort)
    z_top2 = adaptive_snorm(0.6, cohort, top_k=2)  # cohort {0.5,0.7}: mean .6
    assert abs(z_top2) < 1e-9
    assert z_all > z_top2


def test_snorm_rejects_degenerate_cohort():
    with pytest.raises(ValueError):
        adaptive_snorm(0.5, [0.3])
    with pytest.raises(ValueError):
        adaptive_snorm(0.5, [0.3, 0.3, 0.3])


def test_snorm_shrinks_cross_condition_offset():
    rng = np.random.default_rng(10)
    base_t = rng.normal(0.5, 0.05, 200)
    base_c = [list(rng.normal(0.1, 0.05, 30)) for _ in range(200)]
    # condition B: everything shifted +0.3 (different-language shift)
    shift_t = base_t + 0.3
    shift_c = [[x + 0.3 for x in c] for c in base_c]
    raw_gap = abs(np.mean(shift_t) - np.mean(base_t))
    z_gap = abs(np.mean(snorm_scores(shift_t, shift_c))
                - np.mean(snorm_scores(base_t, base_c)))
    assert z_gap < raw_gap / 5


def test_platt_recovers_known_mapping():
    rng = np.random.default_rng(11)
    s = rng.normal(0.0, 2.0, 20000)
    p = 1.0 / (1.0 + np.exp(-(2.0 * s - 1.0)))
    y = (rng.random(20000) < p).astype(int)
    cal = fit_platt(s, y)
    assert abs(cal["a"] - 2.0) < 0.15
    assert abs(cal["b"] - (-1.0)) < 0.15


def test_apply_platt_returns_llrs_centered_by_prior():
    cal = {"a": 1.0, "b": 0.0, "prior_logodds": 0.5}
    out = apply_platt(np.array([0.5, 2.0]), cal)
    assert np.allclose(out, [0.0, 1.5])


def test_platt_requires_both_classes():
    with pytest.raises(ValueError):
        fit_platt(np.array([0.1, 0.2]), np.array([1, 1]))


def test_temperature_recovers_overconfidence_and_lowers_ece():
    rng = np.random.default_rng(12)
    logits = rng.normal(0.0, 2.0, (5000, 5))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in true_p])
    over = np.exp(3.0 * logits)                     # overconfident by T=3
    over = over / over.sum(1, keepdims=True)
    T = fit_temperature(over, y)
    assert 2.0 < T < 4.5
    calibrated = apply_temperature(over, T)
    assert ece(calibrated, y) < ece(over, y)
