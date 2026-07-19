"""Calibration-aware detection metrics. Pure numpy/sklearn — no torch."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

_LOG2 = np.log(2.0)


def cllr(tar_llrs, non_llrs):
    """Log-likelihood-ratio cost (bits). 1.0 = uninformative; lower is better."""
    t = np.asarray(tar_llrs, float)
    n = np.asarray(non_llrs, float)
    if t.size == 0 or n.size == 0:
        raise ValueError("cllr needs both target and non-target LLRs")
    return float(0.5 * (np.mean(np.logaddexp(0.0, -t))
                        + np.mean(np.logaddexp(0.0, n))) / _LOG2)


def pav_llrs(tar_scores, non_scores):
    """PAV (isotonic) optimally-calibrated LLRs; returns targets first."""
    tar = np.asarray(tar_scores, float)
    non = np.asarray(non_scores, float)
    scores = np.concatenate([tar, non])
    labels = np.concatenate([np.ones(tar.size), np.zeros(non.size)])
    p = IsotonicRegression(y_min=0.0, y_max=1.0,
                           out_of_bounds="clip").fit_transform(scores, labels)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    prior = tar.size / scores.size
    return np.log(p / (1.0 - p)) - np.log(prior / (1.0 - prior))


def min_cllr(tar_scores, non_scores):
    """Cllr after PAV optimal calibration: the calibration-free floor."""
    tar = np.asarray(tar_scores, float)
    llrs = pav_llrs(tar_scores, non_scores)
    return cllr(llrs[:tar.size], llrs[tar.size:])


def _rates(tar, non):
    tar = np.sort(np.asarray(tar, float))
    non = np.sort(np.asarray(non, float))
    if tar.size == 0 or non.size == 0:
        raise ValueError("need both target and non-target scores")
    thr = np.unique(np.concatenate([tar, non]))
    p_miss = np.searchsorted(tar, thr, side="left") / tar.size   # P(tar < t)
    p_fa = 1.0 - np.searchsorted(non, thr, side="left") / non.size  # P(non >= t)
    return p_miss, p_fa


def eer(tar_scores, non_scores):
    p_miss, p_fa = _rates(tar_scores, non_scores)
    i = int(np.argmin(np.abs(p_fa - p_miss)))
    return float((p_fa[i] + p_miss[i]) / 2.0)


def min_dcf(tar_scores, non_scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    """Minimum normalized detection cost at the given operating point."""
    p_miss, p_fa = _rates(tar_scores, non_scores)
    dcf = c_miss * p_target * p_miss + c_fa * (1.0 - p_target) * p_fa
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1.0 - p_target)))
