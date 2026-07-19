"""Score-calibration transforms: adaptive s-norm, Platt, temperature scaling."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def adaptive_snorm(score, cohort, top_k=None):
    """(score - mean(top-k cohort)) / std(top-k cohort). AS-norm, test-side."""
    c = np.sort(np.asarray(cohort, float))[::-1]
    if top_k is not None:
        c = c[:top_k]
    if c.size < 2:
        raise ValueError("cohort too small for s-norm (need >= 2 scores)")
    sd = c.std()
    if sd < 1e-8:
        raise ValueError("degenerate cohort: zero variance")
    return float((score - c.mean()) / sd)


def snorm_scores(scores, cohorts, top_k=None):
    return np.array([adaptive_snorm(s, c, top_k)
                     for s, c in zip(scores, cohorts)])


def fit_platt(scores, labels):
    """Logistic (Platt) calibration on 1-D scores. Returns {a, b, prior_logodds}."""
    s = np.asarray(scores, float).reshape(-1, 1)
    y = np.asarray(labels, int)
    if len(np.unique(y)) < 2:
        raise ValueError("Platt fit needs both target and non-target labels")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(s, y)
    prior = float(y.mean())
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]),
            "prior_logodds": float(np.log(prior / (1.0 - prior)))}


def apply_platt(scores, cal):
    """Calibrated LLRs: posterior log-odds minus training prior log-odds."""
    s = np.asarray(scores, float)
    return cal["a"] * s + cal["b"] - cal["prior_logodds"]


def _temp_softmax(posteriors, T):
    logp = np.log(np.clip(np.asarray(posteriors, float), 1e-12, None)) / T
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(posteriors, labels, bounds=(0.05, 20.0)):
    """Single-parameter temperature minimizing NLL on a dev fold."""
    y = np.asarray(labels, int)
    idx = np.arange(y.size)

    def nll(T):
        p = _temp_softmax(posteriors, T)
        return -np.mean(np.log(p[idx, y] + 1e-12))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def apply_temperature(posteriors, T):
    return _temp_softmax(posteriors, T)
