import os
import pytest
from eval.esd import (EMOTIONS, emotion_from_utt, language_from_speaker,
                      parse_converted_name, load_split_basenames)


def test_emotions_order_matches_esd_numbering():
    assert EMOTIONS == ["neutral", "angry", "happy", "sad", "surprise"]


def test_emotion_from_utt_boundaries():
    assert emotion_from_utt(21) == "neutral"
    assert emotion_from_utt(350) == "neutral"
    assert emotion_from_utt(351) == "angry"
    assert emotion_from_utt(700) == "angry"
    assert emotion_from_utt(701) == "happy"
    assert emotion_from_utt(1400) == "sad"
    assert emotion_from_utt(1401) == "surprise"
    assert emotion_from_utt("0011_000844") == "happy"


def test_language_from_speaker():
    assert language_from_speaker("0001") == "zh"
    assert language_from_speaker("0010") == "zh"
    assert language_from_speaker("0011") == "en"
    assert language_from_speaker("0020") == "en"
    with pytest.raises(ValueError):
        language_from_speaker("0021")


def test_parse_converted_name():
    d = parse_converted_name("0011_000021.wav0012_000371.wav")
    assert d == {"source_speaker": "0011", "source_utt": "0011_000021",
                 "target_speaker": "0012", "target_utt": "0012_000371"}
    d2 = parse_converted_name("/some/dir/0011_000021.wav0012_000371.npy")
    assert d2["target_utt"] == "0012_000371"
    with pytest.raises(ValueError):
        parse_converted_name("garbage.wav")


def test_load_split_basenames(tmp_path):
    p = tmp_path / "split.txt"
    p.write_text(
        "{'audio': '/home/x/ESD/val/0014_000716.wav', 'hubert': '1 2', 'duration': 3.9}\n"
        "\n"
        "{'audio': '/home/x/ESD/val/0011_000353.wav', 'hubert': '1', 'duration': 2.2}\n",
        encoding="utf-8")
    assert load_split_basenames(str(p)) == ["0014_000716.wav", "0011_000353.wav"]
