import numpy as np
import pytest
from eval.metrics import cllr, pav_llrs, min_cllr, eer, min_dcf


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
    close_tar = RNG.normal(0.2, 1.0, 2000)
    close_non = RNG.normal(-0.2, 1.0, 2000)
    assert min_cllr(close_tar, close_non) > min_cllr(TAR, NON)


def test_min_dcf_bounds_and_operating_point():
    v = min_dcf(TAR, NON, p_target=0.05)
    assert 0.0 < v <= 1.0
    assert min_dcf(np.array([1.0, 2.0]), np.array([-2.0, -1.0])) == 0.0
