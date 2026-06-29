# ZEST Smoke-Test Reproduction — Design

**Date:** 2026-06-20
**Paper:** Zero Shot Audio to Audio Emotion Transfer With Speaker Disentanglement (ICASSP 2024), arXiv:2401.04511
**Upstream code:** https://github.com/iiscleap/ZEST (mirrored under `C:\ZEST\code`)

## 1. Goal

Get the existing ZEST pipeline to **execute end-to-end on Kaggle** against a **small ESD
(English) subset**, fixing the bugs and hardcoded paths that block a run. This is a
**smoke test**, not a result reproduction.

### Success criteria
- Each of the 5 stages runs to completion without crashing on the subset and writes its
  expected artifacts.
- The run ends by producing **at least one converted `.wav`** in the DSDT setting
  (source content/speaker preserved, target emotion applied).
- A short run report records, per stage: command, runtime, artifact counts, pass/fail.

### Explicit non-goals (out of scope)
- Matching the paper's objective metrics (emotion accuracy, CER, speaker accuracy) or MOS/SMOS.
- Full-length training (F0 predictor 500 epochs, HiFi-GAN 100–150k steps).
- The five non-DSDT conversion settings (SSST/SSDT/DSST/UTE/USS).
- Unseen-emotion (CREMA-D) / unseen-speaker (TIMIT) evaluations and any LDC-licensed data.
- Audio quality — a few-epoch/few-step model will sound bad; that is acceptable.

## 2. Environment & inputs

- **Compute:** Kaggle notebook with GPU (T4/P100). Internet must be enabled for one-time
  HuggingFace/SpeechBrain model downloads.
- **Dataset:** ESD English speakers `0011`–`0020` (user-provided / Kaggle dataset mount,
  read-only). Working copies are written under `/kaggle/working`.
- **Reused shipped artifacts (verified to cover the full ESD English set, no regeneration needed):**
  - `code/f0.pickle` — `dict` of `basename.wav -> F0 ndarray`, 17,500 entries, speakers 0011–0020.
  - `code/train_esd.txt` (15,000), `code/val_esd.txt`, `code/test_esd.txt` — one dict-literal
    per line: `{"audio": <abs path>, "hubert": "<space-separated ints>", "duration": <float>}`.
    Basenames match `f0.pickle` (verified 300/300 sample).
  - `code/esd_f0_stats.pth` — `{"mean", "std"}` F0 normalization stats.
- **Backbones downloaded at runtime (public, no LDC license needed):**
  SpeechBrain `spkrec-ecapa-voxceleb` (x-vectors), HF `wav2vec2-large-robust-ft-swbd-300h`
  (SACE/F0-predictor emotion encoder). HuBERT is **not** needed — tokens are shipped.

## 3. Pipeline dependency graph

```
ESD English wavs (subset)
  │
  ├─ Stage 1 EASE
  │    get_speaker_embedding.py → x-vectors/*.npy        (SpeechBrain ECAPA)
  │    speaker_classifier.py    → EASE.pth, EASE_embeddings/*.npy
  │
  ├─ shipped: subset manifests (from *_esd.txt) + f0.pickle + esd_f0_stats.pth
  │
  ├─ Stage 2 F0 predictor                                 (wav2vec2-large-robust)
  │    pitch_attention_adv.py → f0_predictor.pth          (EASE_embeddings + tokens + f0.pickle + wavs)
  │    pitch_inference.py     → f0_contours/*.npy         (= HiFi-GAN pitch_folder)
  │    get_wav2vec_feats.py   → wav2vec_feats/*.npy       (= HiFi-GAN emo_folder, SACE embeddings)
  │
  ├─ Stage 3 HiFi-GAN
  │    train.py → checkpoints/{g_*,do_*}                  (manifest + f0_contours + wav2vec_feats
  │                                                         + EASE_embeddings + esd_f0_stats)
  │
  └─ Stage 4 Conversion (DSDT)
       pitch_convert.py → pred_DSDT_f0/*.npy
       inference.py --convert → converted .wav            ✅ SUCCESS ARTIFACT
```

## 4. Architecture of the fix (Approach A: overlay + in-place patches)

Keep the upstream structure and each stage independently runnable. Introduce a single
source of truth for paths and smoke-size knobs; patch the hardcoded sites to read from it;
fix portability bugs defensively; drive everything from a Kaggle notebook.

### 4.1 Central config (`code/zest_paths.py`)
A small module resolving all paths and knobs from **environment variables** with
Kaggle-friendly defaults (env vars chosen so they propagate to each stage subprocess
launched from the notebook). Exposes, at minimum:
- Roots: `ZEST_CODE`, `ESD_WAV_DIR` (read-only source), `WORK_DIR` (`/kaggle/working/zest`).
- Derived dirs (created if missing): `data/{train,val,test}`, `x_vectors/`, `EASE_embeddings/`,
  `f0_contours/`, `wav2vec_feats/`, `pred_DSDT_f0/`, `checkpoints/`.
- Shipped artifacts: `F0_PICKLE`, `F0_STATS`, subset manifest paths.
- Smoke knobs: `SMOKE` flag, `UTTS_PER_SPEAKER`, `F0_EPOCHS`, `HIFIGAN_STEPS`,
  `EASE_EPOCHS`, batch sizes.

The notebook prepends `code/` to `PYTHONPATH` so `import zest_paths` works from every
stage directory regardless of `cwd`.

### 4.2 Stage 0 — data prep (new notebook cells / small helper)
1. **Select subset:** all 10 speakers (0011–0020); for each, a few utterances spanning all
   five emotion ranges (file-id buckets: ≤350 neutral, 351–700 angry, 701–1050 happy,
   1051–1400 sad, >1400 surprise). Respect the shipped train/val/test split membership.
   **Concrete default:** 2 train utts/emotion/speaker (≈100 train files total), 1 val and
   1 test utt/emotion/speaker (≈50 each), all tunable via `UTTS_PER_SPEAKER`.
2. **Materialize wavs:** copy selected wavs into `WORK_DIR/data/{train,val,test}` (Kaggle
   input is read-only; EASE and the F0 predictor list these folders).
3. **Generate subset manifests** `{train,val,test}_esd_subset.txt`: filter the shipped
   `*_esd.txt` to subset basenames and **rewrite each `audio` path** to the materialized
   wav location (HiFi-GAN loads audio from the manifest path — this is mandatory).
4. Reuse `f0.pickle` and `esd_f0_stats.pth` unchanged (keyed by basename).

### 4.3 Stage edits (the fix-list)
- **`EASE/get_speaker_embedding.py`** (lines 8–9): replace hardcoded `folder` /
  `target_folder` with `zest_paths` (input = a subset wav folder, output = `x_vectors/`).
- **`EASE/speaker_classifier.py`** (lines 120–126): replace `/folder/to/x-vectors` and the
  train/val/test `/folder/to/...` placeholders with `zest_paths`; lower training epochs to
  `EASE_EPOCHS` (already only 10). `get_embedding()` writes `EASE_embeddings/`.
- **`F0_predictor/config.py`** (lines 1–10): point `train/val/test_datasets` and
  `*_tokens_orig` and `f0_file` at `zest_paths` values (subset folders, subset manifests,
  shipped `f0.pickle`).
- **`F0_predictor/pitch_attention_adv.py`**: fix `getspkrlabel` EASE path (line 84) and the
  `create_dataset` `/folder/to/...` folders (lines 267–274) to use `zest_paths`; reduce the
  hardcoded `range(500)` epochs (line 308) to `F0_EPOCHS`; reduce default batch size.
- **`F0_predictor/pitch_inference.py`, `get_wav2vec_feats.py`, `pitch_convert.py`** (not yet
  line-audited): identify their hardcoded paths during implementation and route them through
  `zest_paths`. Outputs: `f0_contours/`, `wav2vec_feats/`, `pred_DSDT_f0/`.
- **`HiFi-GAN/dataset.py`** (line 291): replace hardcoded `/ZEST/code/EASE/EASE_embeddings/`
  with `zest_paths.EASE_embeddings`; verify `emo_folder`/`pitch_folder` trailing-slash
  string concatenation.
- **`HiFi-GAN/hubert_alladv.json`** → generate a Kaggle variant pointing
  `input_training_file`/`input_validation_file` at the subset manifests and `f0_stats` at the
  shipped stats; pass `--checkpoint_path/--pitch_folder/--emo_folder/--training_steps` via CLI
  (already supported). Set `training_steps` to `HIFIGAN_STEPS`.
- **`HiFi-GAN/inference.py`** (not yet line-audited): route paths via `zest_paths`; run with
  `--convert` for DSDT.

### 4.4 Portability fixes (apply across the repo)
- Replace `import pickle5 as pickle` with stdlib `import pickle` (Python ≥3.8 reads protocol 5).
- Make full-object checkpoints torch-2.6-safe: load with `weights_only=False` (or migrate
  `torch.save(model)`/`torch.load` to `state_dict`); handle `weight_norm` parametrization
  serialization (repo issues #3/#4). Decide minimal vs. state_dict during implementation.
- Guard CUDA/CPU device selection (already mostly present) and `num_gpus=0` single-process path.

### 4.5 Kaggle notebook driver (`kaggle_smoke.ipynb`)
One cell per stage, top to bottom, each printing artifact counts and timing:
0. Setup: `pip install -r requirements.txt`, set env vars / `PYTHONPATH`, enable internet check.
1. Data prep (Stage 0).
2. EASE: `get_speaker_embedding.py` → `speaker_classifier.py`.
3. F0 predictor: `pitch_attention_adv.py` → `pitch_inference.py` → `get_wav2vec_feats.py`.
4. HiFi-GAN train (`HIFIGAN_STEPS` small).
5. Conversion: `pitch_convert.py` → `inference.py --convert`; play/inspect the output wav.
6. Run report.

Cells are independently re-runnable so failures are fixed and retried per stage.

## 5. Iteration loop
Local machine has no GPU, so heavy stages are validated on Kaggle. Workflow: prepare/patch
code here → user runs the relevant notebook cell on Kaggle → user pastes the error/log →
diagnose and patch → repeat. Pure-Python/CPU logic (data prep, manifest rewrite, imports)
can be sanity-checked locally.

## 6. Risks & mitigations
- **torch version drift on Kaggle** (weight_norm parametrization, `weights_only` default) —
  most likely failure; mitigate per 4.4.
- **YAAPT/`amfm_decompy` build** — only needed if `pitch_inference`/`get_wav2vec_feats`/
  `pitch_convert` recompute F0. Prefer reusing `f0.pickle`; if a stage insists on recomputing,
  install `amfm_decompy` (pure-Python) or feed cached F0.
- **Model downloads require Kaggle internet ON** (~1.3 GB wav2vec2 + ~80 MB ECAPA).
- **ESD layout on Kaggle** — confirm English wavs are reachable and named `00NN_NNNNNN.wav`.
- **custom_collate hubert padding uses `max_len_f0`** (`pitch_attention_adv.py:243`) — latent;
  fix only if it crashes on the subset.

## 7. Deliverables
1. `code/zest_paths.py` central config.
2. Patched stage scripts (EASE, F0_predictor, HiFi-GAN) routed through config + portability fixes.
3. Generated Kaggle HiFi-GAN config json + subset manifests.
4. `kaggle_smoke.ipynb` driver.
5. Short README/run-report describing how to run on Kaggle and what success looks like.

## 8. Note on version control
`C:\ZEST` is **not** a git repository, so this design is saved but not committed. Initializing
git is optional and can be done separately before implementation if desired.
