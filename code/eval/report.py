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
        "degradation": (en_cal - zh_cal)
        if (en_cal is not None and zh_cal is not None) else None}
    return {"system": manifest["meta"]["system"],
            "speaker": spk,
            "emotion": {"pooled": _try(emotion_metrics, emo, None, None),
                        "en": _try(emotion_metrics, emo, "en", "en"),
                        "zh": _try(emotion_metrics, emo, "zh", "zh")},
            "cer": cer_metrics(cers)}


def compare_reports(base, cand):
    """Recursive numeric diff: candidate minus baseline. Non-numeric -> None."""
    def walk(b, c):
        if isinstance(b, dict) and isinstance(c, dict):
            return {k: walk(b.get(k), c.get(k)) for k in c if k in b}
        if (isinstance(b, (int, float)) and not isinstance(b, bool)
                and isinstance(c, (int, float)) and not isinstance(c, bool)):
            return round(c - b, 6)
        return None
    return walk(base, cand)


def bootstrap_delta_ci(metric_fn, base_items, cand_items, key_fn,
                       n_boot=200, seed=0, alpha=0.05):
    """Paired percentile-bootstrap CI for metric_fn(cand) - metric_fn(base),
    resampling shared conv_file (key_fn) keys jointly."""
    b_by, c_by = {}, {}
    for it in base_items:
        b_by.setdefault(key_fn(it), []).append(it)
    for it in cand_items:
        c_by.setdefault(key_fn(it), []).append(it)
    keys = sorted(set(b_by) & set(c_by))
    if not keys:
        raise ValueError("no shared keys between baseline and candidate")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        bs = [x for k in ks for x in b_by[k]]
        cs = [x for k in ks for x in c_by[k]]
        deltas.append(metric_fn(cs) - metric_fn(bs))
    return (float(np.percentile(deltas, 100 * alpha / 2)),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))))


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _speaker_min_cllr_stat(trials):
    """Pooled speaker minCllr on a raw speaker-trial list (already eval-filtered)."""
    s, y, coh = _spk_arrays(trials)
    z = calibration.snorm_scores(s, coh, TOP_K)
    return metrics.min_cllr(z[y], z[~y])


def _emotion_accuracy_stat(records):
    """Pooled emotion accuracy on a raw emotion-record list (already eval-filtered)."""
    P, y = _emo_arrays(records)
    return float((P.argmax(axis=1) == y).mean())


def _cer_stat(records):
    """Corpus CER on a raw cer-record list (already eval-/language-filtered)."""
    return metrics.cer_aggregate([(r["ref"], r["hyp"]) for r in records])


def _safe_ci(metric_fn, base_items, cand_items, n_boot, seed):
    try:
        return bootstrap_delta_ci(metric_fn, base_items, cand_items,
                                  key_fn=lambda r: r["conv_file"],
                                  n_boot=n_boot, seed=seed)
    except ValueError:
        return None


def ab_confidence_intervals(base_manifest, cand_manifest, n_boot=1000, seed=0):
    """Paired bootstrap 95% CIs for headline A/B metric deltas (cand - base).

    Resamples shared conv_file keys on the eval fold. Returns {metric_key:
    (lo, hi)}; None where a metric has no shared eval trials."""
    def ev(items):
        return [x for x in items if x["split"] == "eval"]
    out = {
        "speaker.pooled.min_cllr": _safe_ci(
            _speaker_min_cllr_stat, ev(base_manifest["speaker_trials"]),
            ev(cand_manifest["speaker_trials"]), n_boot, seed),
        "emotion.pooled.accuracy": _safe_ci(
            _emotion_accuracy_stat, ev(base_manifest["emotion_records"]),
            ev(cand_manifest["emotion_records"]), n_boot, seed),
    }
    for lang in ("en", "zh"):
        b = [r for r in ev(base_manifest["cer_records"]) if r["language"] == lang]
        c = [r for r in ev(cand_manifest["cer_records"]) if r["language"] == lang]
        out[f"cer.{lang}"] = (_safe_ci(_cer_stat, b, c, n_boot, seed)
                              if (b and c) else None)
    return out


def _fmt_ci(pair):
    if not pair:
        return "—"
    lo, hi = pair
    return f"[{lo:.4f}, {hi:.4f}]"


_SPK_COLS = ["n_trials", "raw_eer", "snorm_eer", "min_cllr", "act_cllr",
             "min_dcf"]
_EMO_COLS = ["n", "accuracy", "ece_raw", "ece_calibrated", "temperature",
             "det_eer", "det_min_cllr", "det_act_cllr"]


def _table(title, cols, rows):
    lines = [f"## {title}", "", "| panel | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, data in rows:
        cells = [_fmt(data.get(c)) if data else "—" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_markdown(cand, base=None, delta=None, ci=None):
    L = [f"# ZEST evaluation report — {cand['system']}", ""]
    panels = [(k, cand["speaker"][k]) for k in ("pooled", "en", "zh")]
    L += _table("Speaker preservation (calibrated verification)",
                _SPK_COLS, panels)
    tr = cand["speaker"]["transfer_en_to_zh"]
    L += ["### Cross-lingual transfer (EN-fit calibration applied to ZH)", "",
          f"- actCllr with EN calibration: {_fmt(tr['act_cllr_en_cal'])}",
          f"- actCllr with ZH calibration: {_fmt(tr['act_cllr_zh_cal'])}",
          f"- degradation (EN-cal − ZH-cal): {_fmt(tr['degradation'])}", ""]
    L += _table("Emotion transfer", _EMO_COLS,
                [(k, cand["emotion"][k]) for k in ("pooled", "en", "zh")])
    L += ["## Textual preservation (CER, per language — never pooled)", ""]
    for lang in ("en", "zh"):
        c = cand["cer"][lang]
        L.append(f"- {lang}: CER {_fmt(c['cer'])} (n={c['n']})")
    L.append("")
    if base is not None and delta is not None:
        L += [f"## A/B vs baseline — {base['system']} → {cand['system']}", "",
              "| metric | baseline | candidate | Δ (cand−base) | 95% CI (Δ) |",
              "|---|---|---|---|---|"]
        for sec, panel, key in (("speaker", "pooled", "min_cllr"),
                                ("speaker", "pooled", "act_cllr"),
                                ("speaker", "pooled", "snorm_eer"),
                                ("emotion", "pooled", "accuracy"),
                                ("emotion", "pooled", "det_act_cllr")):
            b = (base[sec][panel] or {}).get(key)
            c = (cand[sec][panel] or {}).get(key)
            d = ((delta.get(sec) or {}).get(panel) or {}).get(key)
            civ = (ci or {}).get(f"{sec}.{panel}.{key}")
            L.append(f"| {sec}.{panel}.{key} | {_fmt(b)} | {_fmt(c)} | {_fmt(d)} | {_fmt_ci(civ)} |")
        for lang in ("en", "zh"):
            b = base["cer"][lang]["cer"]
            c = cand["cer"][lang]["cer"]
            d = ((delta.get("cer") or {}).get(lang) or {}).get("cer")
            civ = (ci or {}).get(f"cer.{lang}")
            L.append(f"| cer.{lang} | {_fmt(b)} | {_fmt(c)} | {_fmt(d)} | {_fmt_ci(civ)} |")
        L.append("")
    return "\n".join(L)
