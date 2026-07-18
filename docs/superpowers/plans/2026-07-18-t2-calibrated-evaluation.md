# T2 Calibrated Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ZEST's evaluation harness: Stage A (Kaggle) scores converted audio into a JSON scores manifest; Stage B (local, pure Python) applies score calibration (as-norm, Platt, temperature) and reports EER/minCllr/actCllr/minDCF/ECE/CER with EN/ZH panels and first-class A/B comparison.

**Architecture:** Two-stage design from the spec (`docs/superpowers/specs/2026-07-18-t2-calibrated-evaluation-design.md`). Pure modules (`esd.py`, `manifest.py`, `metrics.py`, `calibration.py`, `report.py`) use only numpy/scipy/sklearn and are fully TDD'd locally with synthetic data. Thin GPU scripts (`score_converted.py`, `train_emotion_probe.py`) run on Kaggle and are only `py_compile`-checked locally.

**Tech Stack:** Python 3.10+, numpy, scipy, scikit-learn, pytest (local); torch, torchaudio, transformers, speechbrain, openai/whisper-small via transformers (Kaggle only).

## Global Constraints

- Pure modules (`code/eval/esd.py`, `manifest.py`, `metrics.py`, `calibration.py`, `report.py`, `calibrate_report.py`) MUST NOT import torch, torchaudio, transformers, or speechbrain.
- All tests run from repo root: PowerShell `$env:PYTHONPATH="code"; python -m pytest code/tests -v`.
- Coverage target ≥80% on the five pure modules (not the two Kaggle scripts).
- EN and ZH CER are NEVER pooled into one number.
- All randomness seeded (`np.random.default_rng(seed)`).
- Emotion class order everywhere: `["neutral", "angry", "happy", "sad", "surprise"]` (ESD utterance-number order, mirrors `code/prepare_esd_data.py:10-16`).
- ESD speakers 0001–0010 → `zh`, 0011–0020 → `en`.
- Commit after every task, conventional commits (`feat:`/`test:`/`docs:`), no attribution footer.
- The dev fold comes from `code/val_esd.txt` basenames; the eval fold from `code/test_esd.txt` basenames.

---

### Task 1: Package scaffold + pure ESD helpers (`esd.py`)

**Files:**
- Create: `code/eval/__init__.py` (empty)
- Create: `code/eval/esd.py`
- Create: `code/tests/__init__.py` (empty)
- Test: `code/tests/test_esd.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `EMOTIONS: list[str]` (canonical class order); `emotion_from_utt(utt: str|int) -> str`; `language_from_speaker(spk: str) -> str` ("en"/"zh"); `parse_converted_name(fname: str) -> dict` with keys `source_speaker, source_utt, target_speaker, target_utt` (e.g. `{"source_speaker": "0011", "source_utt": "0011_000021", ...}`); `load_split_basenames(path: str) -> list[str]`.

- [ ] **Step 1: Install local dev deps (idempotent)**

Run: `pip install numpy scipy scikit-learn pytest pytest-cov`
Expected: exits 0.

- [ ] **Step 2: Write the failing tests**

Create `code/tests/test_esd.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_esd.py -v`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'eval'`.

- [ ] **Step 4: Write the implementation**

Create empty `code/eval/__init__.py` and `code/tests/__init__.py`. Create `code/eval/esd.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_esd.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add code/eval/__init__.py code/eval/esd.py code/tests/__init__.py code/tests/test_esd.py
git commit -m "feat: add eval package with pure ESD naming helpers (T2)"
```

---

### Task 2: Scores-manifest schema (`manifest.py`)

**Files:**
- Create: `code/eval/manifest.py`
- Test: `code/tests/test_manifest.py`

**Interfaces:**
- Consumes: `EMOTIONS` from `eval.esd`.
- Produces: `ManifestError(ValueError)`; `new_manifest(system: str, git_commit: str = "", models: dict | None = None) -> dict`; `validate_manifest(m: dict) -> None` (raises `ManifestError`); `save_manifest(m: dict, path: str) -> None` (validates then writes JSON); `load_manifest(path: str) -> dict` (reads then validates). Manifest dict shape: `{"meta": {...}, "speaker_trials": [...], "emotion_records": [...], "cer_records": [...]}` exactly as in spec §3.

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_manifest.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_manifest.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'eval.manifest'`.

- [ ] **Step 3: Write the implementation**

Create `code/eval/manifest.py`:

```python
"""Scores-manifest schema: the Stage A (Kaggle) <-> Stage B (local) contract."""
import json

from eval.esd import EMOTIONS

SCHEMA_VERSION = 1
LANGS = {"en", "zh"}
SPLITS = {"dev", "eval"}

_SPK_KEYS = {"conv_file": str, "enroll_speaker": str, "cosine": float,
             "is_target": bool, "cohort_cosines": list, "language": str,
             "split": str, "setting": str}
_EMO_KEYS = {"conv_file": str, "target_emotion": str, "posterior": dict,
             "language": str, "split": str, "setting": str}
_CER_KEYS = {"conv_file": str, "ref": str, "hyp": str, "language": str,
             "split": str, "setting": str}


class ManifestError(ValueError):
    """Raised when a scores manifest violates the schema."""


def new_manifest(system, git_commit="", models=None):
    return {"meta": {"schema_version": SCHEMA_VERSION, "system": system,
                     "git_commit": git_commit, "models": models or {}},
            "speaker_trials": [], "emotion_records": [], "cer_records": []}


def _check_record(rec, keys, kind, idx):
    if not isinstance(rec, dict):
        raise ManifestError(f"{kind}[{idx}]: expected object, got {type(rec).__name__}")
    for k, t in keys.items():
        if k not in rec:
            raise ManifestError(f"{kind}[{idx}]: missing key {k!r}")
        v = rec[k]
        if t is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ManifestError(f"{kind}[{idx}].{k}: expected number")
        elif not isinstance(v, t):
            raise ManifestError(
                f"{kind}[{idx}].{k}: expected {t.__name__}, got {type(v).__name__}")
    if rec["language"] not in LANGS:
        raise ManifestError(f"{kind}[{idx}].language: {rec['language']!r} not in {sorted(LANGS)}")
    if rec["split"] not in SPLITS:
        raise ManifestError(f"{kind}[{idx}].split: {rec['split']!r} not in {sorted(SPLITS)}")


def validate_manifest(m):
    if not isinstance(m, dict):
        raise ManifestError("manifest must be a dict")
    meta = m.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("system"), str) or not meta["system"]:
        raise ManifestError("meta.system missing or empty")
    for section in ("speaker_trials", "emotion_records", "cer_records"):
        if not isinstance(m.get(section), list):
            raise ManifestError(f"{section} missing or not a list")
    for i, rec in enumerate(m["speaker_trials"]):
        _check_record(rec, _SPK_KEYS, "speaker_trials", i)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   for x in rec["cohort_cosines"]):
            raise ManifestError(f"speaker_trials[{i}].cohort_cosines: non-numeric entry")
    for i, rec in enumerate(m["emotion_records"]):
        _check_record(rec, _EMO_KEYS, "emotion_records", i)
        if rec["target_emotion"] not in EMOTIONS:
            raise ManifestError(f"emotion_records[{i}].target_emotion: {rec['target_emotion']!r}")
        post = rec["posterior"]
        if set(post.keys()) != set(EMOTIONS):
            raise ManifestError(f"emotion_records[{i}].posterior: keys must be {EMOTIONS}")
        total = sum(float(v) for v in post.values())
        if abs(total - 1.0) > 1e-3:
            raise ManifestError(f"emotion_records[{i}].posterior: sum {total:.4f} != 1")
    for i, rec in enumerate(m["cer_records"]):
        _check_record(rec, _CER_KEYS, "cer_records", i)


def save_manifest(m, path):
    validate_manifest(m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    validate_manifest(m)
    return m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_manifest.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add code/eval/manifest.py code/tests/test_manifest.py
git commit -m "feat: add scores-manifest schema with validation (T2)"
```

---

### Task 3: Detection metrics — Cllr, minCllr (PAV), EER, minDCF (`metrics.py` part 1)

**Files:**
- Create: `code/eval/metrics.py`
- Test: `code/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from other eval modules (numpy/sklearn only).
- Produces: `cllr(tar_llrs, non_llrs) -> float`; `pav_llrs(tar_scores, non_scores) -> np.ndarray` (targets first, then non-targets); `min_cllr(tar_scores, non_scores) -> float`; `eer(tar_scores, non_scores) -> float`; `min_dcf(tar_scores, non_scores, p_target=0.05, c_miss=1.0, c_fa=1.0) -> float` (normalized).

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_metrics.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'eval.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `code/eval/metrics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add code/eval/metrics.py code/tests/test_metrics.py
git commit -m "feat: add EER, Cllr/minCllr (PAV), minDCF detection metrics (T2)"
```

---

### Task 4: ECE, CER, bootstrap CI (`metrics.py` part 2)

**Files:**
- Modify: `code/eval/metrics.py` (append)
- Test: `code/tests/test_metrics.py` (append)

**Interfaces:**
- Produces: `ece(posteriors: np.ndarray[N,K], labels: np.ndarray[N], n_bins=10) -> float`; `edit_distance(ref: str, hyp: str) -> int`; `cer(ref: str, hyp: str) -> float`; `cer_aggregate(pairs: list[tuple[str, str]]) -> float` (total edits / total ref chars); `bootstrap_ci(stat_fn, items, n_boot=1000, alpha=0.05, seed=0) -> tuple[float, float]`.

- [ ] **Step 1: Append failing tests to `code/tests/test_metrics.py`**

```python
from eval.metrics import ece, edit_distance, cer, cer_aggregate, bootstrap_ci


def test_ece_zero_when_perfectly_calibrated_and_correct():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 0])
    assert ece(p, y) < 1e-9


def test_ece_high_when_confidently_wrong():
    p = np.array([[0.99, 0.01], [0.99, 0.01]])
    y = np.array([1, 1])
    assert ece(p, y) > 0.9


def test_edit_distance_known_cases():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "") == 3


def test_cer_and_aggregate():
    assert cer("abcd", "abxd") == 0.25
    with pytest.raises(ValueError):
        cer("", "x")
    # aggregate = total edits / total ref chars, NOT mean of per-utt CER
    agg = cer_aggregate([("abcd", "abxd"), ("ab", "ab")])
    assert abs(agg - 1 / 6) < 1e-12


def test_cer_handles_chinese_chars():
    assert cer("你好世界", "你好地界") == 0.25


def test_bootstrap_ci_reproducible_and_covers_mean():
    items = list(RNG.normal(5.0, 1.0, 200))
    stat = lambda xs: float(np.mean(xs))
    lo1, hi1 = bootstrap_ci(stat, items, n_boot=200, seed=7)
    lo2, hi2 = bootstrap_ci(stat, items, n_boot=200, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < 5.0 < hi1
    with pytest.raises(ValueError):
        bootstrap_ci(stat, [], n_boot=10)
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_metrics.py -v`
Expected: ImportError on `ece` — the file-level import failure is the expected RED signal.

- [ ] **Step 3: Append implementation to `code/eval/metrics.py`**

```python
def ece(posteriors, labels, n_bins=10):
    """Expected calibration error (top-label, equal-width bins)."""
    p = np.asarray(posteriors, float)
    y = np.asarray(labels, int)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def edit_distance(ref, hyp):
    """Character-level Levenshtein distance."""
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    if not ref:
        raise ValueError("empty reference transcript")
    return edit_distance(ref, hyp) / len(ref)


def cer_aggregate(pairs):
    """Corpus CER: total edit distance / total reference characters."""
    tot_edits, tot_chars = 0, 0
    for ref, hyp in pairs:
        if not ref:
            raise ValueError("empty reference transcript in pair")
        tot_edits += edit_distance(ref, hyp)
        tot_chars += len(ref)
    if tot_chars == 0:
        raise ValueError("cer_aggregate got no pairs")
    return tot_edits / tot_chars


def bootstrap_ci(stat_fn, items, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for stat_fn over a list of items."""
    n = len(items)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one item")
    rng = np.random.default_rng(seed)
    vals = [stat_fn([items[k] for k in rng.integers(0, n, n)])
            for _ in range(n_boot)]
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_metrics.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add code/eval/metrics.py code/tests/test_metrics.py
git commit -m "feat: add ECE, CER, bootstrap CI metrics (T2)"
```

---

### Task 5: Calibration transforms (`calibration.py`)

**Files:**
- Create: `code/eval/calibration.py`
- Test: `code/tests/test_calibration.py`

**Interfaces:**
- Produces: `adaptive_snorm(score: float, cohort: list[float], top_k: int | None = None) -> float`; `snorm_scores(scores, cohorts, top_k=None) -> np.ndarray`; `fit_platt(scores, labels) -> dict` with keys `a, b, prior_logodds`; `apply_platt(scores, cal: dict) -> np.ndarray` (LLRs); `fit_temperature(posteriors, labels, bounds=(0.05, 20.0)) -> float`; `apply_temperature(posteriors, T) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_calibration.py`:

```python
import numpy as np
import pytest
from eval.calibration import (adaptive_snorm, snorm_scores, fit_platt,
                              apply_platt, fit_temperature, apply_temperature)
from eval.metrics import ece

RNG = np.random.default_rng(0)


def test_snorm_standardizes_against_cohort():
    cohort = [0.0, 0.2, 0.4, 0.6, 0.8]  # mean 0.4
    z = adaptive_snorm(0.4, cohort)
    assert abs(z) < 1e-9
    assert adaptive_snorm(0.9, cohort) > 0


def test_snorm_top_k_uses_highest_cohort_scores():
    cohort = [0.0, 0.0, 0.0, 0.5, 0.7]
    z_all = adaptive_snorm(0.6, cohort)
    z_top2 = adaptive_snorm(0.6, cohort, top_k=2)  # cohort {0.5,0.7}: mean .6
    assert abs(z_top2) < 1e-9
    assert z_all > z_top2


def test_snorm_rejects_degenerate_cohort():
    with pytest.raises(ValueError):
        adaptive_snorm(0.5, [0.3])
    with pytest.raises(ValueError):
        adaptive_snorm(0.5, [0.3, 0.3, 0.3])


def test_snorm_shrinks_cross_condition_offset():
    base_t = RNG.normal(0.5, 0.05, 200)
    base_c = [list(RNG.normal(0.1, 0.05, 30)) for _ in range(200)]
    # condition B: everything shifted +0.3 (different-language shift)
    shift_t = base_t + 0.3
    shift_c = [[x + 0.3 for x in c] for c in base_c]
    raw_gap = abs(np.mean(shift_t) - np.mean(base_t))
    z_gap = abs(np.mean(snorm_scores(shift_t, shift_c))
                - np.mean(snorm_scores(base_t, base_c)))
    assert z_gap < raw_gap / 5


def test_platt_recovers_known_mapping():
    s = RNG.normal(0.0, 2.0, 20000)
    p = 1.0 / (1.0 + np.exp(-(2.0 * s - 1.0)))
    y = (RNG.random(20000) < p).astype(int)
    cal = fit_platt(s, y)
    assert abs(cal["a"] - 2.0) < 0.15
    assert abs(cal["b"] - (-1.0)) < 0.15


def test_apply_platt_returns_llrs_centered_by_prior():
    cal = {"a": 1.0, "b": 0.0, "prior_logodds": 0.5}
    out = apply_platt(np.array([0.5, 2.0]), cal)
    assert np.allclose(out, [0.0, 1.5])


def test_platt_requires_both_classes():
    with pytest.raises(ValueError):
        fit_platt(np.array([0.1, 0.2]), np.array([1, 1]))


def test_temperature_recovers_overconfidence_and_lowers_ece():
    logits = RNG.normal(0.0, 2.0, (5000, 5))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([RNG.choice(5, p=row) for row in true_p])
    over = np.exp(3.0 * logits)                     # overconfident by T=3
    over = over / over.sum(1, keepdims=True)
    T = fit_temperature(over, y)
    assert 2.0 < T < 4.5
    calibrated = apply_temperature(over, T)
    assert ece(calibrated, y) < ece(over, y)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_calibration.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'eval.calibration'`.

- [ ] **Step 3: Write the implementation**

Create `code/eval/calibration.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_calibration.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add code/eval/calibration.py code/tests/test_calibration.py
git commit -m "feat: add s-norm, Platt, temperature calibration transforms (T2)"
```

---

### Task 6: Synthetic-manifest fixture + single-system report (`report.py` part 1)

**Files:**
- Create: `code/tests/conftest.py`
- Create: `code/eval/report.py`
- Test: `code/tests/test_report.py`

**Interfaces:**
- Consumes: `metrics.*`, `calibration.*`, `manifest.new_manifest`, `esd.EMOTIONS`.
- Produces (report.py): `TOP_K = 50`; `speaker_metrics(trials: list[dict], fit_language: str | None, eval_language: str | None) -> dict` with keys `n_trials, raw_eer, snorm_eer, min_cllr, min_dcf, act_cllr` (+ optional `warning`); `emotion_metrics(records, fit_language, eval_language) -> dict` with keys `n, accuracy, confusion, ece_raw, ece_calibrated, temperature, det_eer, det_min_cllr, det_act_cllr` (+ optional `warning`); `cer_metrics(records) -> dict` `{"en": {"n", "cer"}, "zh": {"n", "cer"}}`; `compute_system_report(manifest) -> dict` with keys `system, speaker{pooled,en,zh,transfer_en_to_zh}, emotion{pooled,en,zh}, cer`.
- Produces (conftest.py): `make_synth_manifest(system="synth", seed=0, tar_mu=0.6, acc=0.8) -> dict`.

- [ ] **Step 1: Write the synthetic-manifest helper**

Create `code/tests/conftest.py`:

```python
"""Shared synthetic-manifest builder for Stage B tests. No GPU, no audio."""
import numpy as np

from eval.esd import EMOTIONS
from eval.manifest import new_manifest

SPK_EN = [f"{i:04d}" for i in range(11, 16)]   # 5 EN speakers
SPK_ZH = [f"{i:04d}" for i in range(1, 6)]     # 5 ZH speakers
_REF_EN = "the quick brown fox jumps over the lazy dog"
_REF_ZH = "今天天气很好我们出去玩"


def make_synth_manifest(system="synth", seed=0, tar_mu=0.6, acc=0.8):
    """Synthetic scores manifest: tar_mu = target-cosine mean (speaker quality),
    acc = emotion-classifier accuracy (emotion quality)."""
    rng = np.random.default_rng(seed)
    m = new_manifest(system, git_commit="synthetic")
    for lang, spks, ref in (("en", SPK_EN, _REF_EN), ("zh", SPK_ZH, _REF_ZH)):
        for split in ("dev", "eval"):
            for src in spks:
                for r in range(6):
                    conv = f"conv_{src}_{split}_{r}.wav"
                    for enroll in spks:
                        is_t = enroll == src
                        mu = tar_mu if is_t else 0.1
                        m["speaker_trials"].append({
                            "conv_file": conv, "enroll_speaker": enroll,
                            "cosine": float(rng.normal(mu, 0.1)),
                            "is_target": bool(is_t),
                            "cohort_cosines": [float(x) for x in
                                               rng.normal(0.1, 0.1, 20)],
                            "language": lang, "split": split, "setting": "DSDT"})
                    true_e = int(rng.integers(0, 5))
                    pred_e = true_e if rng.random() < acc else \
                        int((true_e + 1 + rng.integers(0, 4)) % 5)
                    post = {e: 0.075 for e in EMOTIONS}
                    post[EMOTIONS[pred_e]] = 0.7
                    m["emotion_records"].append({
                        "conv_file": conv, "target_emotion": EMOTIONS[true_e],
                        "posterior": post, "language": lang,
                        "split": split, "setting": "DSDT"})
                    hyp = ref if rng.random() < 0.7 else ref[:-2]
                    m["cer_records"].append({
                        "conv_file": conv, "ref": ref, "hyp": hyp,
                        "language": lang, "split": split, "setting": "DSDT"})
    return m
```

- [ ] **Step 2: Write the failing tests**

Create `code/tests/test_report.py`:

```python
import numpy as np
import pytest
from conftest import make_synth_manifest
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_report.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'eval.report'`.

- [ ] **Step 4: Write the implementation**

Create `code/eval/report.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_report.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add code/tests/conftest.py code/eval/report.py code/tests/test_report.py
git commit -m "feat: add single-system calibrated report with EN/ZH panels (T2)"
```

---

### Task 7: A/B comparison + Markdown rendering (`report.py` part 2)

**Files:**
- Modify: `code/eval/report.py` (append)
- Test: `code/tests/test_report.py` (append)

**Interfaces:**
- Produces: `compare_reports(base: dict, cand: dict) -> dict` (same nesting, numeric leaves become cand−base deltas, non-numeric → None); `bootstrap_delta_ci(metric_fn, base_items, cand_items, key_fn, n_boot=200, seed=0, alpha=0.05) -> tuple[float, float]` (paired bootstrap over shared `key_fn` keys); `render_markdown(cand: dict, base: dict | None = None, delta: dict | None = None) -> str`.

- [ ] **Step 1: Append failing tests to `code/tests/test_report.py`**

```python
from eval.report import compare_reports, bootstrap_delta_ci, render_markdown
from eval.metrics import cer_aggregate


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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_report.py -v`
Expected: ImportError on `compare_reports`.

- [ ] **Step 3: Append implementation to `code/eval/report.py`**

```python
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


def render_markdown(cand, base=None, delta=None):
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
              "| metric | baseline | candidate | Δ (cand−base) |",
              "|---|---|---|---|"]
        for sec, panel, key in (("speaker", "pooled", "min_cllr"),
                                ("speaker", "pooled", "act_cllr"),
                                ("speaker", "pooled", "snorm_eer"),
                                ("emotion", "pooled", "accuracy"),
                                ("emotion", "pooled", "det_act_cllr")):
            b = (base[sec][panel] or {}).get(key)
            c = (cand[sec][panel] or {}).get(key)
            d = ((delta.get(sec) or {}).get(panel) or {}).get(key)
            L.append(f"| {sec}.{panel}.{key} | {_fmt(b)} | {_fmt(c)} | {_fmt(d)} |")
        for lang in ("en", "zh"):
            b = base["cer"][lang]["cer"]
            c = cand["cer"][lang]["cer"]
            d = ((delta.get("cer") or {}).get(lang) or {}).get("cer")
            L.append(f"| cer.{lang} | {_fmt(b)} | {_fmt(c)} | {_fmt(d)} |")
        L.append("")
    return "\n".join(L)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_report.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add code/eval/report.py code/tests/test_report.py
git commit -m "feat: add A/B comparison with paired bootstrap and markdown report (T2)"
```

---

### Task 8: Stage B CLI (`calibrate_report.py`) + synthetic end-to-end test

**Files:**
- Create: `code/eval/calibrate_report.py`
- Test: `code/tests/test_cli.py`

**Interfaces:**
- Consumes: `load_manifest`, `compute_system_report`, `compare_reports`, `render_markdown`.
- Produces: `main(argv: list[str] | None = None) -> int`; CLI flags `--candidate PATH` (required), `--baseline PATH`, `--out-json PATH` (default `report.json`), `--out-md PATH` (default `report.md`).

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_cli.py`:

```python
import json
from conftest import make_synth_manifest
from eval.manifest import save_manifest
from eval.calibrate_report import main


def test_single_system_run(tmp_path):
    cand = tmp_path / "cand.json"
    save_manifest(make_synth_manifest("wavlm", seed=8), str(cand))
    oj, om = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(["--candidate", str(cand),
               "--out-json", str(oj), "--out-md", str(om)])
    assert rc == 0
    out = json.loads(oj.read_text(encoding="utf-8"))
    assert out["candidate"]["system"] == "wavlm"
    assert out["baseline"] is None and out["delta"] is None
    assert "# ZEST evaluation report" in om.read_text(encoding="utf-8")


def test_ab_run_produces_delta(tmp_path):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    save_manifest(make_synth_manifest("w2v2", seed=9, tar_mu=0.4), str(base))
    save_manifest(make_synth_manifest("wavlm", seed=9, tar_mu=0.7), str(cand))
    oj, om = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(["--candidate", str(cand), "--baseline", str(base),
               "--out-json", str(oj), "--out-md", str(om)])
    assert rc == 0
    out = json.loads(oj.read_text(encoding="utf-8"))
    assert out["delta"]["speaker"]["pooled"]["min_cllr"] < 0
    assert "A/B" in om.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_cli.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'eval.calibrate_report'`.

- [ ] **Step 3: Write the implementation**

Create `code/eval/calibrate_report.py`:

```python
"""Stage B CLI: scores manifest(s) -> calibrated metrics report.

Usage (repo root):
  PYTHONPATH=code python -m eval.calibrate_report --candidate wavlm.json \
      [--baseline w2v2.json] [--out-json report.json] [--out-md report.md]
"""
import argparse
import json
import sys

from eval.manifest import load_manifest
from eval.report import compare_reports, compute_system_report, render_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True,
                    help="scores manifest of the system under test")
    ap.add_argument("--baseline", default=None,
                    help="optional baseline manifest for A/B comparison")
    ap.add_argument("--out-json", default="report.json")
    ap.add_argument("--out-md", default="report.md")
    args = ap.parse_args(argv)

    cand = compute_system_report(load_manifest(args.candidate))
    base = delta = None
    if args.baseline:
        base = compute_system_report(load_manifest(args.baseline))
        delta = compare_reports(base, cand)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"candidate": cand, "baseline": base, "delta": delta},
                  f, indent=2)
    md = render_markdown(cand, base, delta)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full local suite with coverage**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests -v --cov=eval --cov-report=term-missing`
Expected: all tests pass (≈32); coverage ≥80% on `esd.py`, `manifest.py`, `metrics.py`, `calibration.py`, `report.py`, `calibrate_report.py`.

- [ ] **Step 5: Commit**

```bash
git add code/eval/calibrate_report.py code/tests/test_cli.py
git commit -m "feat: add Stage B CLI producing calibrated eval reports (T2)"
```

---

### Task 9: Stage A extractor (`score_converted.py`, Kaggle-executed)

**Files:**
- Create: `code/eval/score_converted.py`

**Interfaces:**
- Consumes: `eval.esd` helpers, `eval.manifest.new_manifest/save_manifest`, `EmotionProbe` from `eval.train_emotion_probe` (defined in Task 10; `py_compile` does not resolve imports, so writing this import before Task 10 lands is safe).
- Produces: CLI `python -m eval.score_converted --converted-dir D --esd-train-dir D --val-split code/val_esd.txt --test-split code/test_esd.txt --probe emotion_probe.pth --system NAME --out manifest.json [--transcripts-tsv T | --skip-cer] [--setting DSDT] [--enroll-per-spk 20]`.
- NOT unit-tested locally (imports torch); verified with `py_compile` only. Executed on Kaggle.

- [ ] **Step 1: Write the implementation**

Create `code/eval/score_converted.py`:

```python
"""Stage A (Kaggle, GPU): score converted wavs into a scores manifest.

Hard-fails on: empty converted dir, missing transcript (unless --skip-cer),
model-load failure, NaN embeddings — never emits a silently-empty manifest.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torchaudio

from eval.esd import (EMOTIONS, emotion_from_utt, language_from_speaker,
                      load_split_basenames, parse_converted_name)
from eval.manifest import new_manifest, save_manifest
from eval.train_emotion_probe import EmotionProbe

SR = 16000
_WHISPER_LANG = {"en": "english", "zh": "chinese"}


def _die(msg):
    print(f"[score_converted] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_wav(path, device):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(dim=0, keepdim=True).to(device)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_enrollments(esd_train_dir, per_spk, ecapa, device):
    """Mean ECAPA embedding per speaker over its lowest-numbered real wavs
    (lowest ESD ids are neutral utterances)."""
    by_spk = {}
    for p in sorted(glob.glob(os.path.join(esd_train_dir, "*.wav"))):
        spk = os.path.basename(p)[:4]
        by_spk.setdefault(spk, []).append(p)
    if not by_spk:
        _die(f"no real ESD wavs found in {esd_train_dir}")
    enroll = {}
    for spk, paths in sorted(by_spk.items()):
        embs = []
        for p in paths[:per_spk]:
            with torch.no_grad():
                e = ecapa.encode_batch(_load_wav(p, device)).squeeze()
            embs.append(e)
        emb = torch.stack(embs).mean(dim=0)
        if torch.isnan(emb).any():
            _die(f"NaN enrollment embedding for speaker {spk}")
        enroll[spk] = torch.nn.functional.normalize(emb, dim=0)
    return enroll


def load_transcripts(tsv_path):
    refs = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                utt, text = line.split("\t", 1)
                refs[utt.replace(".wav", "")] = text.strip()
    if not refs:
        _die(f"transcripts file {tsv_path} is empty")
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted-dir", required=True)
    ap.add_argument("--esd-train-dir", required=True,
                    help="dir of real ESD wavs named SPKR_UTTNO.wav (enrollment)")
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--probe", required=True, help="emotion_probe.pth")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts-tsv", default=None,
                    help="TSV utt<TAB>text for CER refs")
    ap.add_argument("--skip-cer", action="store_true",
                    help="explicitly skip CER records (no transcripts available)")
    ap.add_argument("--setting", default="DSDT")
    ap.add_argument("--enroll-per-spk", type=int, default=20)
    args = ap.parse_args(argv)

    conv_paths = sorted(glob.glob(os.path.join(args.converted_dir, "*.wav")))
    if not conv_paths:
        _die(f"no converted wavs in {args.converted_dir}")
    if args.transcripts_tsv is None and not args.skip_cer:
        _die("provide --transcripts-tsv or pass --skip-cer explicitly")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # older speechbrain
        from speechbrain.pretrained import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device})

    from transformers import (HubertModel, Wav2Vec2FeatureExtractor,
                              WhisperForConditionalGeneration, WhisperProcessor)
    hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960").to(device).eval()
    probe = EmotionProbe()
    probe.load_state_dict(torch.load(args.probe, map_location=device,
                                     weights_only=True))
    probe.to(device).eval()

    whisper_proc = whisper = refs = None
    if not args.skip_cer:
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        whisper = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small").to(device).eval()
        refs = load_transcripts(args.transcripts_tsv)

    dev_set = set(load_split_basenames(args.val_split))
    eval_set = set(load_split_basenames(args.test_split))
    enroll = build_enrollments(args.esd_train_dir, args.enroll_per_spk,
                               ecapa, device)

    m = new_manifest(args.system, git_commit=_git_commit(),
                     models={"spk": "speechbrain/spkrec-ecapa-voxceleb",
                             "ser": f"hubert-base probe ({args.probe})",
                             "asr": "openai/whisper-small"})
    for path in conv_paths:
        info = parse_converted_name(path)
        src_base = info["source_utt"] + ".wav"
        if src_base in dev_set:
            split = "dev"
        elif src_base in eval_set:
            split = "eval"
        else:
            _die(f"source {src_base} not in val or test split — "
                 f"cannot assign dev/eval fold")
        lang = language_from_speaker(info["source_speaker"])
        wav = _load_wav(path, device)

        with torch.no_grad():
            e = ecapa.encode_batch(wav).squeeze()
        if torch.isnan(e).any():
            _die(f"NaN embedding for {path}")
        e = torch.nn.functional.normalize(e, dim=0)
        cos = {spk: float(torch.dot(e, emb)) for spk, emb in enroll.items()}
        for spk in sorted(enroll):
            cohort = [cos[o] for o in sorted(enroll)
                      if o not in (spk, info["source_speaker"])]
            m["speaker_trials"].append({
                "conv_file": os.path.basename(path), "enroll_speaker": spk,
                "cosine": cos[spk],
                "is_target": spk == info["source_speaker"],
                "cohort_cosines": cohort, "language": lang,
                "split": split, "setting": args.setting})

        with torch.no_grad():
            iv = hubert_fe(wav.squeeze().cpu().numpy(), sampling_rate=SR,
                           return_tensors="pt").input_values.to(device)
            feats = hubert(iv).last_hidden_state.mean(dim=1)
            post = torch.softmax(probe(feats), dim=-1).squeeze().cpu().numpy()
        m["emotion_records"].append({
            "conv_file": os.path.basename(path),
            "target_emotion": emotion_from_utt(info["target_utt"]),
            "posterior": {e_: float(p) for e_, p in zip(EMOTIONS, post)},
            "language": lang, "split": split, "setting": args.setting})

        if not args.skip_cer:
            if info["source_utt"] not in refs:
                _die(f"missing transcript for {info['source_utt']}")
            with torch.no_grad():
                feats_w = whisper_proc(wav.squeeze().cpu().numpy(),
                                       sampling_rate=SR,
                                       return_tensors="pt").input_features
                forced = whisper_proc.get_decoder_prompt_ids(
                    language=_WHISPER_LANG[lang], task="transcribe")
                ids = whisper.generate(feats_w.to(device),
                                       forced_decoder_ids=forced)
            hyp = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]
            ref = refs[info["source_utt"]]
            if lang == "en":
                ref = "".join(ch for ch in ref.lower() if ch.isalnum() or ch == " ")
                hyp = "".join(ch for ch in hyp.lower() if ch.isalnum() or ch == " ")
            m["cer_records"].append({
                "conv_file": os.path.basename(path), "ref": ref,
                "hyp": hyp.strip(), "language": lang,
                "split": split, "setting": args.setting})

    save_manifest(m, args.out)
    print(f"[score_converted] wrote {args.out}: "
          f"{len(m['speaker_trials'])} speaker trials, "
          f"{len(m['emotion_records'])} emotion records, "
          f"{len(m['cer_records'])} cer records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it compiles (no unit tests — torch glue)**

Run: `python -m py_compile code/eval/score_converted.py`
Expected: exits 0, no output. (Import would fail locally without torch — `py_compile` checks syntax only, which is the local verification bar for Kaggle scripts, same as T1.)

- [ ] **Step 3: Verify the pure suite still passes (no accidental torch import)**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests -q`
Expected: all tests still pass.

- [ ] **Step 4: Commit**

```bash
git add code/eval/score_converted.py
git commit -m "feat: add Stage A Kaggle extractor producing scores manifests (T2)"
```

---

### Task 10: Emotion-probe trainer, Kaggle runbook, docs update

**Files:**
- Create: `code/eval/train_emotion_probe.py`
- Create: `code/eval/KAGGLE_EVAL.md`
- Modify: `task.md` (T2 status line)
- Modify: `progress.md` (T2 section + "Next up")

**Interfaces:**
- Produces: `EmotionProbe(torch.nn.Module)` — `__init__(self, in_dim=768, hidden=256, n_classes=5)`, `forward(self, x) -> logits [B, 5]` (imported by `score_converted.py`); CLI `python -m eval.train_emotion_probe --esd-wav-dir D --train-split code/train_esd.txt --val-split code/val_esd.txt --out emotion_probe.pth [--epochs 30]`.

- [ ] **Step 1: Write the probe trainer**

Create `code/eval/train_emotion_probe.py`:

```python
"""One-time (Kaggle, GPU): train the independent 5-class ESD emotion probe.

Backbone: frozen facebook/hubert-base-ls960, mean-pooled last hidden state —
deliberately NOT the WavLM family SACE uses, so the evaluator is independent
of the system under test (spec §4.2). Labels come free from ESD numbering.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from eval.esd import EMOTIONS, emotion_from_utt, load_split_basenames

SR = 16000


class EmotionProbe(nn.Module):
    """One-hidden-layer MLP on 768-d mean-pooled HuBERT-base features."""

    def __init__(self, in_dim=768, hidden=256, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def _die(msg):
    print(f"[train_emotion_probe] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def extract_features(wav_dir, basenames, device):
    import torchaudio
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960").to(device).eval()
    X, y = [], []
    for base in basenames:
        path = os.path.join(wav_dir, base)
        if not os.path.exists(path):
            continue
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(dim=0)
        with torch.no_grad():
            iv = fe(wav.numpy(), sampling_rate=SR,
                    return_tensors="pt").input_values.to(device)
            feat = hubert(iv).last_hidden_state.mean(dim=1).squeeze().cpu()
        X.append(feat.numpy())
        y.append(EMOTIONS.index(emotion_from_utt(base)))
    if not X:
        _die(f"no wavs from the split found in {wav_dir}")
    return np.stack(X), np.array(y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd-wav-dir", required=True)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--out", default="emotion_probe.pth")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = extract_features(args.esd_wav_dir,
                                load_split_basenames(args.train_split), device)
    Xva, yva = extract_features(args.esd_wav_dir,
                                load_split_basenames(args.val_split), device)
    print(f"train {len(ytr)} utts, val {len(yva)} utts")

    probe = EmotionProbe().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)

    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        probe.train()
        perm = torch.randperm(len(ytr_t), device=device)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(probe(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            acc = float((probe(Xva_t).argmax(1) == yva_t).float().mean())
        print(f"epoch {epoch + 1}: val acc {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone()
                          for k, v in probe.state_dict().items()}

    if best_acc < 0.5:
        _die(f"probe val accuracy {best_acc:.3f} < 0.5 — evaluator unusable")
    torch.save(best_state, args.out)
    print(f"saved {args.out} (best val acc {best_acc:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify compile + suite**

Run: `python -m py_compile code/eval/train_emotion_probe.py`
Expected: exits 0.
Run: `$env:PYTHONPATH="code"; python -m pytest code/tests -q`
Expected: all pass.

- [ ] **Step 3: Write the Kaggle runbook**

Create `code/eval/KAGGLE_EVAL.md` with exactly this content (note: the inner
code fences use ~~~ so they nest inside this plan; write them as ``` in the
actual file):

~~~markdown
# T2 evaluation on Kaggle — runbook

Prereqs: ZEST repo cloned at /kaggle/working/ZEST on the branch under test,
ESD subset prepared (as in kaggle_smoke.ipynb), converted wavs present.

## Cell A — deps + path

```python
%pip -q install speechbrain transformers torchaudio
import sys, os
sys.path.insert(0, "/kaggle/working/ZEST/code")
os.chdir("/kaggle/working/ZEST")
```

## Cell B — build the transcripts TSV from ESD's per-speaker txt files

```python
import glob
with open("transcripts.tsv", "w", encoding="utf-8") as out:
    for txt in glob.glob("/kaggle/input/**/00*.txt", recursive=True):
        for line in open(txt, encoding="utf-8", errors="replace"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                out.write(f"{parts[0]}\t{parts[1]}\n")
```

## Cell C — train the emotion probe (one-time; reuse the .pth afterwards)

```python
!PYTHONPATH=code python -m eval.train_emotion_probe \
  --esd-wav-dir ESD_subset/all_wavs --train-split code/train_esd.txt \
  --val-split code/val_esd.txt --out emotion_probe.pth
```

## Cell D — score each system's converted wavs

```python
!PYTHONPATH=code python -m eval.score_converted \
  --converted-dir converted_w2v2 --esd-train-dir ESD_subset/all_wavs \
  --val-split code/val_esd.txt --test-split code/test_esd.txt \
  --probe emotion_probe.pth --transcripts-tsv transcripts.tsv \
  --system w2v2-baseline --out manifest_w2v2.json
!PYTHONPATH=code python -m eval.score_converted \
  --converted-dir converted_wavlm --esd-train-dir ESD_subset/all_wavs \
  --val-split code/val_esd.txt --test-split code/test_esd.txt \
  --probe emotion_probe.pth --transcripts-tsv transcripts.tsv \
  --system wavlm-T1 --out manifest_wavlm.json
```

(If transcripts are unavailable in the subset, replace --transcripts-tsv
with --skip-cer.)

## Cell E — report (also runnable locally after downloading the manifests)

```python
!PYTHONPATH=code python -m eval.calibrate_report \
  --candidate manifest_wavlm.json --baseline manifest_w2v2.json \
  --out-json t1_vs_baseline.json --out-md t1_vs_baseline.md
```

Download the two manifest_*.json files — Stage B runs locally from them.
~~~

- [ ] **Step 4: Update `task.md` and `progress.md`**

In `task.md`, change:

```
### T2 — Calibrated evaluation (as-norm + EER/Cllr)  ·  Status: TODO
```
to
```
### T2 — Calibrated evaluation (as-norm + EER/Cllr)  ·  Status: CODE COMPLETE (pending Kaggle scoring runs). Spec: docs/superpowers/specs/2026-07-18-t2-calibrated-evaluation-design.md
```

In `progress.md`:
- In "Status at a glance", change the `T2–T5` row to two rows: `T2 — Calibrated eval harness (code) | ✅ Code complete — pending Kaggle scoring` and `T3–T5 — Remaining optimizations (see task.md) | ⬜ Not started`.
- Append a `## T2 — Calibrated evaluation harness (✅ code complete 2026-07-18, pending Kaggle runs)` section stating: the repo had NO converted-audio evaluation (paper metrics unimplemented); built the two-stage harness (`code/eval/`, spec + plan in `docs/superpowers/`); pure core (esd/manifest/metrics/calibration/report + CLI) fully unit-tested locally (report the actual test count and coverage from Task 8 Step 4); Stage A (`score_converted.py`) + probe trainer (`train_emotion_probe.py`) are py_compile-verified, pending execution per `code/eval/KAGGLE_EVAL.md`.
- Update "Next up" to: 1. Kaggle run — T1 retrain, then T2 probe training + scoring of baseline & WavLM systems, then Stage B A/B report; 2. T3.

- [ ] **Step 5: Full suite + commit**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests -q`
Expected: all pass.

```bash
git add code/eval/train_emotion_probe.py code/eval/KAGGLE_EVAL.md task.md progress.md
git commit -m "feat: add emotion-probe trainer, Kaggle eval runbook, T2 docs (T2)"
```

---

## Execution notes

- Task order is strict for 1–8 (each imports the previous); Task 9 additionally
  imports `EmotionProbe` from Task 10's file — but both are py_compile-only, and
  `py_compile` does not resolve imports, so either order works; both must exist
  before the Kaggle run.
- If sklearn's `LogisticRegression`/`IsotonicRegression` versions shift numerics,
  the tests use loose tolerances deliberately — do not tighten them.
- Windows note: all local test commands are PowerShell (`$env:PYTHONPATH="code"`);
  the `git`/`pip` commands work in either shell.
