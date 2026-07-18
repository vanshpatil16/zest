# T2 — Calibrated Evaluation Harness — Design

**Date:** 2026-07-18
**Status:** Approved (brainstorming session)
**Branch context:** work will build on `feat/wavlm-sace` history; harness is model-agnostic
**Companion docs:** `task.md` (T2 entry), `progress.md` (M3 research findings)

## 1. Problem & goal

ZEST currently has **no evaluation of converted audio at all**. The paper (Sec. 4.3.1)
defines three objective metrics — emotion-conversion accuracy, CER, speaker accuracy —
but none are implemented in this repo; the only `accuracy`/`f1_score` calls in `code/`
are training-time diagnostics of internal classifier heads.

T2 therefore delivers a **full calibrated evaluation harness**: base scorers for all
three quality dimensions **plus** the score-calibration layer (as-norm, Platt,
temperature scaling) and calibration-aware metrics (EER, minCllr/actCllr, minDCF, ECE),
with first-class **A/B comparison** between two systems (headline use: wav2vec2
baseline vs. WavLM/T1 after the Kaggle retrain) and **per-language (EN/ZH) reporting**
that quantifies the language-deviation problem for T3/T4.

Why calibration: raw cosine scores are not comparable across conditions and have no
stable threshold — a threshold tuned on English silently breaks on Mandarin. The
minCllr↔actCllr gap measures exactly how trustworthy the reported numbers are.

## 2. Constraints

- **No GPU or ESD data locally.** Anything touching audio/models runs on Kaggle
  (`kaggle_smoke.ipynb` flow). Local work is limited to pure-Python code + tests.
- ESD is bilingual: speakers 0001–0010 Mandarin, 0011–0020 English. Language labels
  derive from speaker ID for free.
- Converted wavs are produced by the existing pipeline (`pitch_convert.py` →
  HiFi-GAN inference) with filenames encoding source/target utterance IDs.
- Follow repo conventions; new code lives under `code/eval/`, tests under `code/tests/`.

## 3. Architecture (Approach 1 — two-stage with a scores-manifest contract)

Chosen over (2) a monolithic Kaggle script — untestable locally, re-extracts on every
calibration tweak — and (3) wholesale SpeechBrain-VoxCeleb/BOSARIS adoption — recipes
assume standard SV trials, not converted-audio + 5-class emotion + CER. We instead
**port the specific formulas** (EER, PAV-based minCllr, minDCF, Platt) into a small
owned, tested core (~150 lines of math).

```
code/eval/
  manifest.py           # scores-manifest schema + load/save/validate      (pure)
  calibration.py        # adaptive s-norm, Platt scaling, temperature scaling (pure)
  metrics.py            # EER, minCllr/actCllr (PAV), minDCF, ECE, CER agg,
                        #   bootstrap CIs                                   (pure)
  report.py             # A/B deltas + significance, Markdown/JSON render   (pure)
  score_converted.py    # Stage A: converted wavs -> manifest   [Kaggle/GPU]
  train_emotion_probe.py# one-time: train 5-class ESD SER probe [Kaggle/GPU]
  calibrate_report.py   # Stage B: 1-2 manifests -> report      [local/CPU]
code/tests/
  test_metrics.py  test_calibration.py  test_manifest.py  test_report.py
```

“Pure” modules use only numpy/scipy/sklearn — no torch, no audio — so all calibration
research is developed and unit-tested locally against synthetic manifests.

### Data flow

```
converted wavs + real ESD refs ──► Stage A (Kaggle): score_converted.py
    ├─ ECAPA embeddings → cosine vs every speaker enrollment → speaker trials
    ├─ 5-class ESD probe SER → emotion posterior             → emotion records
    └─ Whisper (lang forced per utt) → hyp text → CER inputs → cer records
                        │
                        ▼
        scores manifest (JSON), one per system (baseline.json, candidate.json)
                        │
                        ▼
Stage B (local): calibrate_report.py
    ├─ fit calibration on dev fold, apply to disjoint eval fold
    ├─ EER / minCllr / actCllr / minDCF / ECE / CER
    ├─ EN-only, ZH-only, pooled + EN→ZH transfer panel
    └─ A/B deltas + bootstrap CIs ──► report.md + report.json
```

A/B comparison = Stage B on two manifests; single-system scoring is the one-manifest
degenerate case.

### Manifest schema (the contract)

```jsonc
{
  "meta": { "system": "wavlm-large-T1", "git_commit": "...", "created": "...",
            "models": { "spk": "...", "ser": "...", "asr": "..." } },
  "speaker_trials": [
    { "conv_file": "...", "enroll_speaker": "0011", "cosine": 0.63,
      "is_target": true, "cohort_cosines": [/* top-k cohort scores for s-norm */],
      "language": "en", "split": "eval", "setting": "DSDT" } ],
  "emotion_records": [
    { "conv_file": "...", "target_emotion": "angry",
      "posterior": { "neutral": 0.05, "happy": 0.10, "sad": 0.05,
                     "angry": 0.70, "surprise": 0.10 },
      "language": "en", "split": "eval", "setting": "DSDT" } ],
  "cer_records": [
    { "conv_file": "...", "ref": "...", "hyp": "...",
      "language": "zh", "split": "eval", "setting": "DSDT" } ]
}
```

One converted wav yields N speaker trials (one per enrolled speaker: 1 target +
N−1 non-target), one emotion record, one CER record.

## 4. Metric & calibration design

### 4.1 Speaker preservation (verification framing, fully calibrated)
- Enrollment embedding per speaker = mean ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)
  embedding over that speaker's **real** ESD utterances.
- Each converted wav (intended source speaker known) is scored against **every**
  enrollment → full target/non-target trial list.
- Calibration pipeline: **adaptive s-norm** (cohort = enrollment set, top-k) →
  **Platt** (logistic) calibration fit on dev → **EER, minCllr, actCllr, minDCF**
  on eval. This is the textbook SV calibration case named in `task.md`.

### 4.2 Emotion transfer (multiclass, calibrated)
- Evaluator: **5-class probe (one-hidden-layer MLP) on frozen SSL features**
  (mean-pooled `facebook/hubert-base-ls960` hidden states — already used in the
  pipeline for content units, so no new download; crucially **not** the WavLM
  family SACE uses, keeping the evaluator independent of the system under test).
  Trained once on real ESD audio (`train_emotion_probe.py`, weights saved as
  `emotion_probe.pth` artifact). Chosen over the off-the-shelf IEMOCAP 4-class
  SER (no `surprise` → undercounts an ESD emotion).
- Primary: **accuracy** (argmax == target), **confusion matrix**, **macro-ECE** with
  **temperature scaling** fit on dev.
- Additional detection view: score = `P(target emotion)` → same EER/minCllr machinery
  as speaker (shared code path, near-zero extra cost) — satisfies task.md's
  EER/Cllr-for-emotion ask.

### 4.3 Textual preservation (CER, uncalibrated)
- ASR: **Whisper multilingual** (`openai/whisper-small`; `-base` as OOM fallback),
  language **forced per-utterance** from the known EN/ZH label.
- CER vs. real ESD transcript of the source utterance: ZH on characters; EN
  lowercased, punctuation-stripped. Reported as plain per-language error rate;
  **EN and ZH CER are never pooled**.

### 4.4 Dev/eval discipline & the trust story
- Calibration params (Platt, temperature, s-norm cohort stats) are **fit on dev,
  applied to a disjoint eval fold**, derived deterministically from the existing
  `code/val_esd.txt` / `code/test_esd.txt` splits.
- Headline number: the **minCllr (calibration floor) vs. actCllr (post-calibration)**
  gap = the calibration-quality measure.

### 4.5 Cross-lingual panel (bridge to T3/T4)
- Every metric reported **EN-only / ZH-only / pooled**.
- **Transfer panel:** fit threshold + calibration on **EN**, apply unchanged to
  **ZH**, report the actCllr/EER degradation → quantifies "a threshold tuned on
  English silently breaks on Mandarin".

### 4.6 A/B comparison (first-class)
- Stage B accepts `--baseline b.json --candidate c.json`; reports every metric
  side-by-side with **deltas + bootstrap CIs** (resampling trials/records with a
  fixed seed); accuracy deltas additionally get a paired test on shared trial keys.

## 5. Error handling (fail fast at boundaries)

Stage A (Kaggle):
- Hard-fail (non-zero exit, loud message) on: empty converted-audio dir (M1's
  silent-zero-wavs lesson), missing ESD transcript, model-load failure, NaN/empty
  embedding, un-derivable language label.

Stage B (local):
- Schema-validate manifests on load; reject malformed rows with row-level context.
- Empty dev fold → **degrade to minCllr-only with a loud warning**, never report a
  bogus actCllr.
- Single-manifest mode skips A/B cleanly. EN/ZH CER never silently pooled.

## 6. Testing (TDD, all local, no GPU)

- `test_metrics.py` — EER on two Gaussians with known analytic crossover; PAV output
  monotone; **minCllr ≤ actCllr invariant**; minDCF at a known operating point;
  bootstrap CI reproducible under fixed seed.
- `test_calibration.py` — Platt recovers a known logit→prob map; temperature scaling
  reduces ECE on synthetic over-confident posteriors; as-norm shrinks
  cross-condition score variance.
- `test_manifest.py` — schema validation rejects malformed input; save/load
  round-trip lossless.
- `test_report.py` — A/B delta sign/magnitude; single-system path renders; EN→ZH
  transfer panel computed as specified.
- Coverage target: **≥80 %** on the four pure modules. `score_converted.py` /
  `train_emotion_probe.py` are thin model glue, exercised on Kaggle, not unit-tested.

## 7. Deliverables & sequencing

1. Pure core, TDD: `manifest.py`, `metrics.py`, `calibration.py`, `report.py` +
   4 test files (local, immediately verifiable).
2. `calibrate_report.py` CLI (local; runnable end-to-end on synthetic manifests).
3. Stage A scripts: `train_emotion_probe.py`, `score_converted.py` (py_compile +
   review locally; execute on Kaggle).
4. Kaggle wiring: notebook cells to train the probe, score baseline + WavLM systems,
   and pull manifests back for local Stage B.

Out of scope for T2: subjective MOS/SMOS, CREMA-D/TIMIT zero-shot settings (UTE/USS),
PLDA backend (cosine only for now), and any change to training code.
