import numpy as np
import pytest
from tests.conftest import make_synth_manifest
from eval.report import (speaker_metrics, emotion_metrics, cer_metrics,
                         compute_system_report, compare_reports, bootstrap_delta_ci,
                         render_markdown)
from eval.metrics import cer_aggregate


M = make_synth_manifest(seed=1)


def test_speaker_metrics_shape_and_sanity():
    out = speaker_metrics(M["speaker_trials"], None, None)
    assert out["n_trials"] > 0
    for k in ("raw_eer", "snorm_eer", "min_cllr", "act_cllr", "min_dcf"):
        assert isinstance(out[k], float)
    assert 0.0 <= out["raw_eer"] < 0.5          # well-separated synth scores
    assert out["min_cllr"] <= out["act_cllr"] + 1e-9


def test_speaker_metrics_empty_dev_degrades_loudly():
    eval_only = [t for t in M["speaker_trials"] if t["split"] == "eval"]
    out = speaker_metrics(eval_only, None, None)
    assert out["act_cllr"] is None
    assert "dev" in out["warning"]


def test_speaker_metrics_no_eval_raises():
    dev_only = [t for t in M["speaker_trials"] if t["split"] == "dev"]
    with pytest.raises(ValueError):
        speaker_metrics(dev_only, None, None)


def test_emotion_metrics_accuracy_tracks_synth_quality():
    good = make_synth_manifest(seed=2, acc=0.9)["emotion_records"]
    bad = make_synth_manifest(seed=2, acc=0.4)["emotion_records"]
    g = emotion_metrics(good, None, None)
    b = emotion_metrics(bad, None, None)
    assert g["accuracy"] > b["accuracy"]
    assert len(g["confusion"]) == 5 and len(g["confusion"][0]) == 5
    assert g["det_min_cllr"] <= g["det_act_cllr"] + 1e-9
    assert isinstance(g["temperature"], float)


def test_cer_metrics_never_pools_languages():
    out = cer_metrics(M["cer_records"])
    assert set(out.keys()) == {"en", "zh"}
    assert out["en"]["cer"] is not None
    assert out["zh"]["cer"] is not None
    assert "pooled" not in out


def test_compute_system_report_structure():
    rep = compute_system_report(M)
    assert rep["system"] == "synth"
    assert set(rep["speaker"].keys()) == {"pooled", "en", "zh",
                                          "transfer_en_to_zh"}
    assert set(rep["emotion"].keys()) == {"pooled", "en", "zh"}
    tr = rep["speaker"]["transfer_en_to_zh"]
    assert "act_cllr_en_cal" in tr and "degradation" in tr


def test_compare_reports_deltas():
    base = compute_system_report(make_synth_manifest("base", seed=3, tar_mu=0.4))
    cand = compute_system_report(make_synth_manifest("cand", seed=3, tar_mu=0.7))
    d = compare_reports(base, cand)
    # candidate separates speakers better -> minCllr goes DOWN
    assert d["speaker"]["pooled"]["min_cllr"] < 0
    assert d["speaker"]["pooled"]["n_trials"] == 0


def test_bootstrap_delta_ci_paired_and_signed():
    base = make_synth_manifest("base", seed=4)["cer_records"]
    cand = [dict(r, hyp=r["ref"]) for r in base]     # candidate: perfect ASR
    fn = lambda rs: cer_aggregate([(r["ref"], r["hyp"]) for r in rs])
    lo, hi = bootstrap_delta_ci(fn, base, cand, key_fn=lambda r: r["conv_file"],
                                n_boot=50, seed=5)
    assert hi <= 0.0                                  # CER strictly improves
    with pytest.raises(ValueError):
        bootstrap_delta_ci(fn, base, [dict(r, conv_file="other.wav")
                                      for r in cand],
                           key_fn=lambda r: r["conv_file"], n_boot=5)


def test_render_markdown_single_and_ab():
    cand = compute_system_report(M)
    md = render_markdown(cand)
    assert "# ZEST evaluation report" in md
    assert "transfer" in md.lower()
    base = compute_system_report(make_synth_manifest("base", seed=6))
    md2 = render_markdown(cand, base, compare_reports(base, cand))
    assert "A/B" in md2 and "base" in md2
