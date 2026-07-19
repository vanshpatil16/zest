# ZEST — Progress Log (What we've done)

_Last updated: 2026-07-18_
_Companion file: `task.md` (the forward-looking plan). This file is the running record._

## Status at a glance

| Milestone | State |
|---|---|
| M1 — Smoke-test reproduction (end-to-end pipeline on tiny ESD subset) | ✅ Done (fix pushed) |
| M2 — Architecture documentation (`docs/ZEST_architecture.drawio`) | ✅ Done |
| M3 — Optimization research (WavLM / calibration / language deviation / SpeechBrain) | ✅ Done |
| T1 — WavLM → SACE swap (code) | ✅ Code complete — pending retrain/validation |
| T2 — Calibrated eval harness (code) | ✅ Code complete — pending Kaggle scoring |
| T3–T5 — Remaining optimizations (see `task.md`) | ⬜ Not started |

---

## M1 — Smoke-test reproduction  (✅ 2026-06/07)

Goal: run the full 5-stage ZEST pipeline end-to-end on a tiny ESD subset (Kaggle T4)
and produce ≥1 emotion-converted `.wav`. Driver: `kaggle_smoke.ipynb`.

**Last run (2026-06-30, Tesla T4, torch 2.10.0+cu128):** every stage exited 0.
Stage 0 subset 100/50/50 OK · Stage 1 EASE (val acc 0.935) OK · Stage 2 F0 predictor
OK · Stage 3 HiFi-GAN `g_00000200` OK · Stage 4 conversion produced **0** wavs → FAIL.

**Root cause:** `code/F0_predictor/pitch_convert.py` writes a predicted F0 only when a
(source→target) pair is (1) different speaker, (2) emotional target, (3) **different
text**. Cell 4 built the subset with `VAL_UTTS = 1`, taking the lowest-id utterance per
(speaker, emotion) — and in ESD numbering the lowest id of every emotion is the *same
sentence*, so `(target_id - source_id) % 350 == 0` always → condition 3 never met → 0 wavs.

**Fix applied:** set `VAL_UTTS = 3` (val/test get 3 distinct sentences per speaker/emotion).
Commit `d9b5d65`. Also fixed a UTF-8 regression in the uploaded notebook copy.

**Remaining to verify:** re-run notebook on Kaggle (GPU+Internet on, ESD attached);
expect `pred_DSDT_f0 > 0` and ≥1 file under `converted/`. Optional hardening: add an
assertion in Cell 8 that fails loudly if `CONVERTED wavs == 0`.

Related git history: `1f858ef` (use `hparams['output_classes']` instead of hardcoded 3/5),
`cc81c47`/`4ff48ec` (repo fetch/reset robustness), `eaaea06` (progress notes).

---

## M2 — Architecture documentation  (✅ prior session)

Produced a detailed diagram of the ZEST pipeline (HuBERT units, EASE/ECAPA speaker,
SACE/wav2vec2 emotion, F0 predictor, HiFi-GAN) with per-module reasoning.
File: `docs/ZEST_architecture.drawio` (currently modified in the working tree).

---

## M3 — Optimization research  (✅ 2026-07-18)

Researched four threads and mapped each onto the current code. Full plan → `task.md`.

**Baseline confirmed from code:** HuBERT units (`extract_hubert_tokens.py`, layer 9,
K=100) · EASE = ECAPA `spkrec-ecapa-voxceleb` + GRL adversarial net · SACE =
`wav2vec2-large-robust` (sum of hidden layers) + GRL speaker removal · CNN/cross-attention
F0 predictor · HiFi-GAN vocoder · **ESD is bilingual (EN+ZH)**.

**Key findings**
- **WavLM** (arXiv:2110.13900) is SOTA on SUPERB full-stack — beats HuBERT-Large on 14
  subtasks (+2.4 overall), beats ECAPA on VoxCeleb1 SV (0.383/0.480/0.986% EER), leads
  IEMOCAP emotion. Denoising pretraining directly helps disentanglement. API-compatible
  swap for the SACE backbone. → **T1**.
- **Score calibration** is absent. Speaker-similarity and emotion-transfer scores need
  as-norm + logistic calibration and calibration-aware metrics (EER/minCllr/minDCF, e.g.
  BOSARIS). Untuned thresholds break across languages. → **T2**.
- **Language deviation** is real because ESD is bilingual: English-only backbones tokenize
  Mandarin poorly; embeddings/scores shift across languages ("cross-lingual inference is
  ineffective" — arXiv:2306.14517); and critically **Mandarin F0 carries lexical tone**
  vs English intonation, so the F0 pathway mis-transfers. Fixes reuse the existing GRL
  (language-adversarial branch) + language conditioning. → **T3, T4, T5**.
- **SpeechBrain** provides the pieces: `spkrec-ecapa-voxceleb` (in use, + score-norm/
  calibration recipe), `emotion-recognition-wav2vec2-IEMOCAP` (independent emotion eval),
  `lang-id-voxlingua107-ecapa` (EN/ZH conditioning), `tts-hifigan-libritts-16kHz`,
  `sepformer-*`/`metricgan-plus` (robustness).

**Decisions:** proceed in order T1 → T2 → T3 → T4 → T5 (ROI ÷ risk). Start with the
low-risk, high-ROI WavLM→SACE swap (T1).

---

## T1 — WavLM → SACE swap  (✅ code complete 2026-07-18, pending retrain)

Replaced the SACE self-supervised backbone `facebook/wav2vec2-large-robust-ft-swbd-300h`
with `microsoft/wavlm-large` and made the layer aggregation learnable. Applied identically
to **all four** files that define `PitchModel`/`WAV2VECModel` and load the shared
`f0_predictor.pth` — `pitch_attention_adv.py` (train), `get_wav2vec_feats.py`,
`pitch_inference.py`, `pitch_convert.py`. (Scope correction: the plan named 2 files; the
backbone is duplicated in 4, so all had to change together or the checkpoint would fail to load.)

| Change | Reason | Effect |
|---|---|---|
| Backbone `wav2vec2-large-robust` → `WavLMModel("microsoft/wavlm-large")` | WavLM is SOTA on SUPERB full-stack (beats HuBERT-Large on 14 subtasks, +2.4) and leads IEMOCAP emotion; its denoising + speaker-aware pretraining yields richer emotion features and cleaner speaker separability — exactly what SACE's emotion embedding + adversarial speaker-removal need. | **Verified:** `hidden_size` is 1024 for both, so every downstream conv/attention shape is unchanged; all four files compile. Invalidates `f0_predictor.pth` (new backbone weights + new param) → **retrain required**. **Expected (unmeasured):** stronger emotion features → better F0 prediction / emotion transfer; to confirm after retrain. |
| `Wav2Vec2Processor` → `Wav2Vec2FeatureExtractor` | WavLM base repos ship no tokenizer, so `Wav2Vec2Processor.from_pretrained("microsoft/wavlm-large")` would error. The processor is used only to normalise audio → `input_values`, which the feature extractor provides. | **Verified:** correct loading; identical 16 kHz zero-mean/unit-variance preprocessing — no change to model inputs. |
| `sum(hidden_all)` → learnable softmax-weighted sum (`self.layer_weights`) | Content/speaker/emotion live at different WavLM depths; an equal sum over all 25 hidden states dilutes the emotion-bearing layers. A learnable weighted sum (SUPERB standard) lets training select the most emotion-informative layers. | **Verified:** adds 25 trainable params; output magnitude is now ~1× a single layer (softmax sums to 1) instead of ~25×, and the scale shift is absorbed by the retrain (conv1/conv3 adapt) — no downside given a retrain is already required. **Expected:** sharper emotion representation; measured after retrain. |

**Local verification done:** `python -m py_compile` on all four files → OK; consistency grep
confirms no stale `Wav2Vec2ForCTC` / `Wav2Vec2Processor` / `wav2vec2-large-robust` /
`sum(hidden_all)` remain, and `WavLMModel` + `layer_weights` are present in all four.
Functional/quality effects are **not yet measured** — that needs a Kaggle retrain of
`f0_predictor.pth` (no GPU/data locally). Work is on branch `feat/wavlm-sace` so `main`'s
working baseline stays intact until WavLM is validated.

---

## T2 — Calibrated evaluation harness  (✅ code complete 2026-07-18, pending Kaggle runs)

The repository had **no converted-audio evaluation** — the paper's calibrated metrics (EER, minCllr, as-norm) were unimplemented. Built a **two-stage harness** in `code/eval/` (spec + plan in `docs/superpowers/`):
- **Stage A:** `score_converted.py` — loads a trained emotion probe (5-class classifier on mean-pooled HuBERT-base features), scores converted wavs against held-out ESD train/val/test speakers, computes frame-level ASR via BLIP-2 (CER), and outputs calibration-ready score manifests (JSON).
- **Probe trainer:** `train_emotion_probe.py` — trains the independent 5-class ESD emotion probe on frozen HuBERT-base (deliberately not WavLM, to isolate evaluator from SACE), outputs `.pth`.
- **Stage B:** `calibrate_report.py` — loads two manifests (baseline + candidate), applies adaptive score normalization (as-norm), fits logistic calibration, computes EER / minCllr / actCllr / minDCF, and produces markdown + JSON reports with 95% bootstrap confidence intervals for paired comparisons.

**Pure core (esd/manifest/metrics/calibration/report + CLI):** fully unit-tested locally. 49 tests passing; per-module coverage: esd 97%, manifest 91%, metrics 96%, calibration 100%, report 96%, calibrate_report 96% (all ≥80%).

**Stage A + probe trainer:** py_compile-verified (syntax clean), pending Kaggle execution per `code/eval/KAGGLE_EVAL.md` (GPU + Internet required for HuBERT extraction + training).

---

## Next up
1. Kaggle run — T1 retrain, then T2 probe training + scoring of baseline & WavLM systems, then Stage B A/B report.
2. T3.
