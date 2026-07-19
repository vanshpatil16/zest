"""Stage B reporting: per-language calibrated metrics + A/B comparison."""
import numpy as np

from eval import calibration, metrics
from eval.esd import EMOTIONS

TOP_K = 50                                    # s-norm cohort top-k
_PRIOR_LOGODDS_5CLASS = float(np.log(0.2 / 0.8))


def _subset(items, language, split):
    return [x for x in items
            if (language is None or x["language"] == language)
            and (split is None or x["split"] == split)]


def _spk_arrays(trials):
    s = np.array([t["cosine"] for t in trials], float)
    y = np.array([bool(t["is_target"]) for t in trials])
    coh = [t["cohort_cosines"] for t in trials]
    return s, y, coh


def speaker_metrics(trials, fit_language, eval_language):
    """S-norm + Platt fit on the dev fold of fit_language, scored on the
    eval fold of eval_language. language=None means pooled."""
    dev = _subset(trials, fit_language, "dev")
    ev = _subset(trials, eval_language, "eval")
    if not ev:
        raise ValueError("no eval-fold speaker trials for this selection")
    s_ev, y_ev, coh_ev = _spk_arrays(ev)
    z_ev = calibration.snorm_scores(s_ev, coh_ev, TOP_K)
    out = {"n_trials": len(ev),
           "raw_eer": metrics.eer(s_ev[y_ev], s_ev[~y_ev]),
           "snorm_eer": metrics.eer(z_ev[y_ev], z_ev[~y_ev]),
           "min_cllr": metrics.min_cllr(z_ev[y_ev], z_ev[~y_ev]),
           "min_dcf": metrics.min_dcf(z_ev[y_ev], z_ev[~y_ev])}
    if dev:
        s_d, y_d, coh_d = _spk_arrays(dev)
        z_d = calibration.snorm_scores(s_d, coh_d, TOP_K)
        cal = calibration.fit_platt(z_d, y_d.astype(int))
        llr = calibration.apply_platt(z_ev, cal)
        out["act_cllr"] = metrics.cllr(llr[y_ev], llr[~y_ev])
    else:
        out["act_cllr"] = None
        out["warning"] = "empty dev fold: actCllr unavailable (minCllr-only)"
    return out


def _emo_arrays(records):
    P = np.array([[r["posterior"][e] for e in EMOTIONS] for r in records], float)
    y = np.array([EMOTIONS.index(r["target_emotion"]) for r in records], int)
    return P, y


def _det_scores(P, y):
    """Flatten to detection trials: each record x 5 classes -> 1 target + 4 non."""
    mask = np.zeros_like(P, bool)
    mask[np.arange(y.size), y] = True
    return P[mask], P[~mask]


def emotion_metrics(records, fit_language, eval_language):
    dev = _subset(records, fit_language, "dev")
    ev = _subset(records, eval_language, "eval")
    if not ev:
        raise ValueError("no eval-fold emotion records for this selection")
    P, y = _emo_arrays(ev)
    pred = P.argmax(axis=1)
    conf = [[int(((pred == j) & (y == i)).sum()) for j in range(5)]
            for i in range(5)]
    tar, non = _det_scores(P, y)
    out = {"n": len(ev), "accuracy": float((pred == y).mean()),
           "confusion": conf, "ece_raw": metrics.ece(P, y),
           "det_eer": metrics.eer(tar, non),
           "det_min_cllr": metrics.min_cllr(tar, non)}
    if dev:
        Pd, yd = _emo_arrays(dev)
        T = calibration.fit_temperature(Pd, yd)
        Pc = calibration.apply_temperature(P, T)
        logodds = (np.log(np.clip(Pc, 1e-12, None))
                   - np.log(np.clip(1.0 - Pc, 1e-12, None)))
        llr = logodds - _PRIOR_LOGODDS_5CLASS
        ltar, lnon = _det_scores(llr, y)
        out.update({"temperature": T,
                    "ece_calibrated": metrics.ece(Pc, y),
                    "det_act_cllr": metrics.cllr(ltar, lnon)})
    else:
        out.update({"temperature": None, "ece_calibrated": None,
                    "det_act_cllr": None,
                    "warning": "empty dev fold: calibrated emotion metrics unavailable"})
    return out


def cer_metrics(records):
    """Per-language corpus CER on the eval fold. EN and ZH are never pooled."""
    out = {}
    for lang in ("en", "zh"):
        rs = _subset(records, lang, "eval")
        if rs:
            out[lang] = {"n": len(rs),
                         "cer": metrics.cer_aggregate(
                             [(r["ref"], r["hyp"]) for r in rs])}
        else:
            out[lang] = {"n": 0, "cer": None}
    return out


def _try(fn, *args):
    try:
        return fn(*args)
    except ValueError:
        return None


def compute_system_report(manifest):
    trials = manifest["speaker_trials"]
    emo = manifest["emotion_records"]
    cers = manifest["cer_records"]
    spk = {"pooled": _try(speaker_metrics, trials, None, None),
           "en": _try(speaker_metrics, trials, "en", "en"),
           "zh": _try(speaker_metrics, trials, "zh", "zh")}
    en_on_zh = _try(speaker_metrics, trials, "en", "zh")
    zh_cal = spk["zh"]["act_cllr"] if spk["zh"] else None
    en_cal = en_on_zh["act_cllr"] if en_on_zh else None
    spk["transfer_en_to_zh"] = {
        "act_cllr_en_cal": en_cal,
        "act_cllr_zh_cal": zh_cal,
        "eer_en_cal": en_on_zh["snorm_eer"] if en_on_zh else None,
        "degradation": (en_cal - zh_cal)
        if (en_cal is not None and zh_cal is not None) else None}
    return {"system": manifest["meta"]["system"],
            "speaker": spk,
            "emotion": {"pooled": _try(emotion_metrics, emo, None, None),
                        "en": _try(emotion_metrics, emo, "en", "en"),
                        "zh": _try(emotion_metrics, emo, "zh", "zh")},
            "cer": cer_metrics(cers)}
