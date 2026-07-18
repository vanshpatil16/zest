"""Pure ESD naming/metadata helpers. No torch, no audio I/O."""
import ast
import os
import re

# ESD utterance-number -> emotion. Mirrors prepare_esd_data.py:10-16 (dataset constant).
EMOTION_RANGES = [
    (0, 350, "neutral"),
    (351, 700, "angry"),
    (701, 1050, "happy"),
    (1051, 1400, "sad"),
    (1401, 99999, "surprise"),
]
EMOTIONS = ["neutral", "angry", "happy", "sad", "surprise"]

_UTT_RE = re.compile(r"(\d{4})_(\d{6})")


def emotion_from_utt(utt):
    """Emotion name for an ESD utterance ('0011_000844', '000844', or int)."""
    n = int(str(utt).split("_")[-1].split(".")[0])
    for lo, hi, name in EMOTION_RANGES:
        if lo <= n <= hi:
            return name
    raise ValueError(f"utterance id out of ESD range: {utt!r}")


def language_from_speaker(spk):
    n = int(spk)
    if 1 <= n <= 10:
        return "zh"
    if 11 <= n <= 20:
        return "en"
    raise ValueError(f"unknown ESD speaker: {spk!r}")


def parse_converted_name(fname):
    """Parse converted-output names like '0011_000021.wav0012_000371.wav'.

    ZEST concatenates source then target utterance names (pitch_convert.py:214).
    """
    ids = _UTT_RE.findall(os.path.basename(fname))
    if len(ids) != 2:
        raise ValueError(f"cannot parse converted filename: {fname!r}")
    (s_spk, s_utt), (t_spk, t_utt) = ids
    return {"source_speaker": s_spk, "source_utt": f"{s_spk}_{s_utt}",
            "target_speaker": t_spk, "target_utt": f"{t_spk}_{t_utt}"}


def load_split_basenames(path):
    """Basenames of the 'audio' field from a ZEST split file (dict-literal lines)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(os.path.basename(ast.literal_eval(line)["audio"]))
    return out
