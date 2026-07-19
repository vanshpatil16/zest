# ZEST — Task Plan (What we're doing & why)

_Last updated: 2026-07-18_
_Companion file: `progress.md` (what has been done). This file is the forward-looking plan._

## Objective

ZEST = **Zero-Shot Emotion Style Transfer** (audio→audio emotion conversion with
speaker disentanglement, ICASSP 2024). The base pipeline runs end-to-end. The goal
of this phase is to **optimize the architecture** using four research threads:
**WavLM**, **score calibration**, the **language-deviation problem**, and
**SpeechBrain models**. Findings are recorded in `progress.md` (Research phase).

## Current architecture (baseline we're improving)

| Stage | Model / file | Detail |
|---|---|---|
| Content units | `facebook/hubert-base-ls960` — `code/extract_hubert_tokens.py:14` | layer 9, K=100 k-means |
| **EASE** speaker | `speechbrain/spkrec-ecapa-voxceleb` — `code/EASE/get_speaker_embedding.py:7` | 192-d x-vector → GRL adversarial net (`code/EASE/speaker_classifier.py:84-117`) → 128-d emotion-agnostic |
| **SACE** emotion | `facebook/wav2vec2-large-robust-ft-swbd-300h` — `code/F0_predictor/pitch_attention_adv.py:150-151` | **sum** of all hidden layers → CNN + GRL speaker removal |
| F0 predictor | CNN + cross-attention fusion, adversarial pitch | `code/F0_predictor/pitch_attention_adv.py` |
| Vocoder | HiFi-GAN | unit + F0 + EASE + SACE → wav |
| Data | **ESD** (bilingual: 10 English + 10 Mandarin speakers, 5 emotions) | — |

Two facts drive the plan: (1) the **gradient-reversal (GRL) machinery already exists**
in EASE and SACE, so adversarial upgrades are cheap; (2) **ESD is bilingual**, which
makes "language deviation" a real, not theoretical, problem.

## Work items (ordered by ROI ÷ risk)

### T1 — WavLM into SACE (+ learnable layer weighting)  ·  Status: CODE COMPLETE (branch `feat/wavlm-sace`; pending Kaggle retrain). Scope note: applied to all 4 `PitchModel` files, not 2.
- **What:** Replace `Wav2Vec2ForCTC(...wav2vec2-large-robust...)` with
  `WavLMModel.from_pretrained("microsoft/wavlm-large")` in
  `code/F0_predictor/pitch_attention_adv.py:150-151` and
  `code/F0_predictor/get_wav2vec_feats.py:150-151`. Replace the plain
  `sum(hidden_all)` (`get_wav2vec_feats.py:83`) with a **learnable softmax-weighted
  sum** over layers (SUPERB style). Use the feature extractor `WavLMModel`, not `...ForCTC`.
- **Why:** WavLM is SOTA on SUPERB full-stack (beats HuBERT-Large on 14 subtasks,
  +2.4 overall); its denoising pretraining directly aids emotion/speaker separation,
  and it leads on IEMOCAP emotion. API-compatible (16 kHz, `output_hidden_states`) → low risk.
- **Effort/Risk:** Low / Low. Lighter alt for T4 smoke test: `microsoft/wavlm-base-plus`.

### T2 — Calibrated evaluation (as-norm + EER/Cllr)  ·  Status: CODE COMPLETE (pending Kaggle scoring runs). Spec: docs/superpowers/specs/2026-07-18-t2-calibrated-evaluation-design.md
- **What:** Add score calibration to the two eval metrics — speaker-similarity
  (cosine/PLDA) and emotion-transfer accuracy. Use **adaptive score normalization
  (as-norm)** + logistic (Platt) calibration; report **EER + minCllr/actCllr + minDCF**
  via BOSARIS or SpeechBrain's VoxCeleb score-norm/calibration recipe. Optionally track
  **ECE / temperature scaling** on the EASE speaker head and SACE emotion adversary.
- **Why:** Raw cosine scores aren't comparable across conditions and have no stable
  threshold — a threshold tuned on English silently breaks on Mandarin. Calibration
  makes reported numbers trustworthy and cross-lingual-honest. Training is unchanged.
- **Effort/Risk:** Low / Low (evaluation-only).

### T3 — Language-adversarial GRL branch on EASE + SACE  ·  Status: TODO
- **What:** Add a **language classifier (EN vs ZH)** behind a gradient-reversal layer
  on both EASE and SACE, reusing `ReverseLayerF` (`code/EASE/speaker_classifier.py:84-117`).
  Labels are free from ESD speaker IDs.
- **Why:** Forces **language-invariant** speaker and emotion embeddings; directly
  attacks language deviation (same pattern as arXiv:2603.08092). ~30 lines, reuses
  existing adversarial machinery.
- **Effort/Risk:** Medium / Low.

### T4 — Language-conditioned F0 / tonal fix  ·  Status: TODO
- **What:** Condition the F0 predictor (and HiFi-GAN) on **language ID**
  (`speechbrain/lang-id-voxlingua107-ecapa`), or train language-specific F0 heads.
- **Why:** Mandarin F0 carries **lexical tone** (changes word meaning); English F0
  carries intonation. An F0 model trained mostly on English intonation mangles Mandarin
  tone — language deviation hiding in the pitch pathway. Highest-value Mandarin
  correctness fix.
- **Effort/Risk:** Medium / Medium.

### T5 — Multilingual backbone + WavLM speaker/units  ·  Status: TODO (do last)
- **What:** (a) Move content/emotion backbones to multilingual SSL
  (`facebook/w2v-bert-2.0`, `facebook/wav2vec2-xls-r-300m`, or mHuBERT).
  (b) Optionally move EASE speaker to `microsoft/wavlm-base-plus-sv`.
  (c) Optionally move content units to WavLM (**requires re-training k-means + HiFi-GAN**).
- **Why:** English-only backbones tokenize Mandarin poorly; multilingual predict+denoise
  pretraining is the right recipe (WavLabLM, arXiv:2309.15317). Highest retraining cost,
  so sequenced last.
- **Effort/Risk:** High / Medium.

## SpeechBrain components in play

| Stage | Model | Use |
|---|---|---|
| Speaker (EASE) | `speechbrain/spkrec-ecapa-voxceleb` *(in use)* | recipe also provides PLDA + score-norm + calibration (T2) |
| Emotion eval | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | independent SER to measure emotion-transfer success |
| Language deviation | `speechbrain/lang-id-voxlingua107-ecapa` | EN/ZH conditioning for T4 |
| Vocoder ref | `speechbrain/tts-hifigan-libritts-16kHz` | maintained 16 kHz HiFi-GAN baseline |
| Robustness | `speechbrain/sepformer-*`, `metricgan-plus-voicebank` | front-end denoise (or rely on WavLM's built-in denoising) |

## Key references
- WavLM: arXiv:2110.13900 · SUPERB/WavLM SV & emotion numbers
- Calibration: BOSARIS / arXiv:2409.05032 · SpeechBrain VoxCeleb recipe
- Language deviation: TidyVoice 2026 (2601.21960) · GRL SV (2603.08092) · spoof threshold transfer (2603.02364) · cross-lingual SER (2306.14517) · WavLabLM (2309.15317)
