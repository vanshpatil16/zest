import numpy as np
import pytest
from tests.conftest import make_synth_manifest
from eval.report import (speaker_metrics, emotion_metrics, cer_metrics,
                         compute_system_report)


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
