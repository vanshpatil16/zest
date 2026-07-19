import numpy as np
import pytest
from eval.metrics import cllr, pav_llrs, min_cllr, eer, min_dcf, ece, edit_distance, cer, cer_aggregate, bootstrap_ci


RNG = np.random.default_rng(42)
TAR = RNG.normal(1.0, 1.0, 4000)   # scores treated as LLRs where relevant
NON = RNG.normal(-1.0, 1.0, 4000)


def test_eer_two_gaussians_analytic():
    # Equal-variance gaussians at ±1: EER = Phi(-1) ≈ 0.1587
    assert abs(eer(TAR, NON) - 0.1587) < 0.02


def test_eer_perfect_separation_is_zero():
    assert eer(np.array([1.0, 2.0]), np.array([-2.0, -1.0])) == 0.0


def test_cllr_of_zero_llrs_is_one():
    z = np.zeros(100)
    assert abs(cllr(z, z) - 1.0) < 1e-12


def test_cllr_requires_both_classes():
    with pytest.raises(ValueError):
        cllr(np.array([]), np.array([0.0]))


def test_pav_llrs_monotone_in_score():
    llrs = pav_llrs(TAR, NON)
    scores = np.concatenate([TAR, NON])
    order = np.argsort(scores)
    diffs = np.diff(llrs[order])
    assert (diffs >= -1e-9).all()


def test_min_cllr_leq_act_cllr():
    # Scores ARE plausible LLRs here, so cllr(TAR, NON) is an actCllr.
    assert min_cllr(TAR, NON) <= cllr(TAR, NON) + 1e-9


def test_min_cllr_degrades_with_overlap():
    rng = np.random.default_rng(13)
    close_tar = rng.normal(0.2, 1.0, 2000)
    close_non = rng.normal(-0.2, 1.0, 2000)
    assert min_cllr(close_tar, close_non) > min_cllr(TAR, NON)


def test_min_dcf_bounds_and_operating_point():
    v = min_dcf(TAR, NON, p_target=0.05)
    assert 0.0 < v <= 1.0
    assert min_dcf(np.array([1.0, 2.0]), np.array([-2.0, -1.0])) == 0.0


def test_ece_zero_when_perfectly_calibrated_and_correct():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 0])
    assert ece(p, y) < 1e-9


def test_ece_high_when_confidently_wrong():
    p = np.array([[0.99, 0.01], [0.99, 0.01]])
    y = np.array([1, 1])
    assert ece(p, y) > 0.9


def test_edit_distance_known_cases():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "") == 3


def test_cer_and_aggregate():
    assert cer("abcd", "abxd") == 0.25
    with pytest.raises(ValueError):
        cer("", "x")
    # aggregate = total edits / total ref chars, NOT mean of per-utt CER
    agg = cer_aggregate([("abcd", "abxd"), ("ab", "ab")])
    assert abs(agg - 1 / 6) < 1e-12


def test_cer_handles_chinese_chars():
    assert cer("你好世界", "你好地界") == 0.25


def test_bootstrap_ci_reproducible_and_covers_mean():
    rng = np.random.default_rng(14)
    items = list(rng.normal(5.0, 1.0, 200))
    stat = lambda xs: float(np.mean(xs))
    lo1, hi1 = bootstrap_ci(stat, items, n_boot=200, seed=7)
    lo2, hi2 = bootstrap_ci(stat, items, n_boot=200, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < 5.0 < hi1
    with pytest.raises(ValueError):
        bootstrap_ci(stat, [], n_boot=10)
