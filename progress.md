# ZEST — Progress Log (What we've done)

_Last updated: 2026-07-18_
_Companion file: `task.md` (the forward-looking plan). This file is the running record._

## Status at a glance

| Milestone | State |
|---|---|
| M1 — Smoke-test reproduction (end-to-end pipeline on tiny ESD subset) | ✅ Done (fix pushed) |
| M2 — Architecture documentation (`docs/ZEST_architecture.drawio`) | ✅ Done |
| M3 — Optimization research (WavLM / calibration / language deviation / SpeechBrain) | ✅ Done |
| T1–T5 — Architecture optimizations (see `task.md`) | ⬜ Not started |

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

## Next up
1. **T1** — WavLM into SACE + learnable layer weighting (see `task.md`).
2. Verify M1 conversion output on Kaggle (`pred_DSDT_f0 > 0`).
