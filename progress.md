# ZEST — Progress / Smoke-Test Status

_Last updated: 2026-07-01_

## Goal
Run the full 5-stage ZEST emotion-voice-conversion pipeline end-to-end on a tiny
ESD subset (Kaggle T4 GPU) and produce **at least one emotion-converted `.wav`**.
Driver notebook: `kaggle_smoke.ipynb` (repo) / `zest-smoke.ipynb` (uploaded copy).

## Latest run (2026-06-30, Kaggle Tesla T4, torch 2.10.0+cu128)

| Stage | Artifact | Count | Status |
|------|----------|------:|--------|
| 0 — subset build | train / val / test wavs | 100 / 50 / 50 | OK |
| 1 — EASE | x-vectors / EASE embeddings | 200 / 200 | OK (val acc 0.935) |
| 2 — F0 predictor | f0_contours / wav2vec_feats | 200 / 200 | OK |
| 3 — HiFi-GAN | checkpoint `g_00000200` | 1 | OK |
| 4 — conversion | **pred_DSDT_f0** | **0** | FAIL |
| Goal | **CONVERTED wavs** | **0** | **FAIL** |

Every stage exited `0` (no crashes). The pipeline is wired correctly and all model
stages work — but the final conversion produced no output.

## Root cause
`code/F0_predictor/pitch_convert.py` writes a predicted F0 only when a
(source -> target) pair satisfies **all three** conditions:

1. different speaker  — `source[:5] not in target_name`
2. emotional target   — `labels[0] > 0`
3. **different text**  — `(target_id - source_id) % 350 != 0`

Cell 4 built the subset with `VAL_UTTS = 1`, taking the **first** (lowest-id)
utterance in each `(speaker, emotion)` bucket. In ESD numbering the lowest id of
every emotion is the **same sentence** (neutral 1, angry 351, happy 701, ...), so
`(target_id - source_id) % 350 == 0` for every candidate. Condition 3 is never
met -> `pred_DSDT_f0 = 0` -> HiFi-GAN inference has nothing to convert -> 0 wavs.

## Fix applied
- Set `VAL_UTTS = 3` in Cell 1 (val/test now get 3 distinct sentences per
  speaker/emotion, so the "different text" condition can be satisfied).
- Commit: `d9b5d65 — fix: bump VAL_UTTS to 3 so DSDT conversion produces output in smoke test`
  (pushed to `origin/main`).
- Also updated the uploaded copy `C:\Users\hp\Downloads\zest-smoke (1).ipynb`
  (fixed a UTF-8 encoding regression introduced during editing; both notebooks
  verified: em-dash intact, valid JSON).

## Next steps
1. Re-upload / re-run the notebook on Kaggle (GPU on, Internet on, ESD dataset
   attached).
2. Re-run **Cell 4** (rebuilds the larger subset: test grows 50 -> ~150 wavs).
3. Re-run **Cells 5-6** if inference reports a missing `.npy` (regenerates
   embeddings / wav2vec feats for the new test files) — otherwise optional.
4. Re-run **Cell 8**; expect `pred_DSDT_f0 > 0` and at least one file under
   `converted/`.
5. Verify Cell 9 report shows non-zero `pred_DSDT_f0` and `CONVERTED wavs`.

## Open items / ideas
- Optional hardening: relax condition 3 in `pitch_convert.py` so even a
  1-utterance subset always yields a converted wav (not needed once `VAL_UTTS=3`,
  and it would weaken the different-text evaluation semantics).
- Consider a tiny assertion at the end of Cell 8 that fails loudly if
  `CONVERTED wavs == 0`, to catch regressions early.
