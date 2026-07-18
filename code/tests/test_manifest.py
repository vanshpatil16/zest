import copy
import pytest
from eval.manifest import (ManifestError, new_manifest, validate_manifest,
                           save_manifest, load_manifest)


def _spk_trial(**over):
    d = {"conv_file": "0011_000021.wav0012_000371.wav", "enroll_speaker": "0011",
         "cosine": 0.63, "is_target": True, "cohort_cosines": [0.1, 0.2],
         "language": "en", "split": "eval", "setting": "DSDT"}
    d.update(over)
    return d


def _emo_record(**over):
    d = {"conv_file": "0011_000021.wav0012_000371.wav", "target_emotion": "angry",
         "posterior": {"neutral": 0.05, "angry": 0.70, "happy": 0.10,
                       "sad": 0.05, "surprise": 0.10},
         "language": "en", "split": "eval", "setting": "DSDT"}
    d.update(over)
    return d


def _cer_record(**over):
    d = {"conv_file": "0011_000021.wav0012_000371.wav", "ref": "hello there",
         "hyp": "hello hare", "language": "en", "split": "eval", "setting": "DSDT"}
    d.update(over)
    return d


def _valid():
    m = new_manifest("sys-a", git_commit="abc123", models={"spk": "ecapa"})
    m["speaker_trials"].append(_spk_trial())
    m["emotion_records"].append(_emo_record())
    m["cer_records"].append(_cer_record())
    return m


def test_valid_manifest_passes():
    validate_manifest(_valid())  # must not raise


def test_missing_key_rejected():
    m = _valid()
    del m["speaker_trials"][0]["cosine"]
    with pytest.raises(ManifestError, match="cosine"):
        validate_manifest(m)


def test_bad_language_rejected():
    m = _valid()
    m["cer_records"][0]["language"] = "fr"
    with pytest.raises(ManifestError, match="language"):
        validate_manifest(m)


def test_bad_split_rejected():
    m = _valid()
    m["speaker_trials"][0]["split"] = "train"
    with pytest.raises(ManifestError, match="split"):
        validate_manifest(m)


def test_posterior_keys_must_cover_emotions():
    m = _valid()
    del m["emotion_records"][0]["posterior"]["surprise"]
    with pytest.raises(ManifestError, match="posterior"):
        validate_manifest(m)


def test_posterior_must_sum_to_one():
    m = _valid()
    m["emotion_records"][0]["posterior"]["angry"] = 0.95
    with pytest.raises(ManifestError, match="sum"):
        validate_manifest(m)


def test_unknown_target_emotion_rejected():
    m = _valid()
    m["emotion_records"][0]["target_emotion"] = "bored"
    with pytest.raises(ManifestError, match="target_emotion"):
        validate_manifest(m)


def test_save_load_round_trip(tmp_path):
    m = _valid()
    p = str(tmp_path / "m.json")
    save_manifest(m, p)
    m2 = load_manifest(p)
    assert m2 == m


def test_load_rejects_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"meta": {"system": "x"}}', encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(str(p))
