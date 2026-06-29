# Running ZEST on Kaggle GPU — Steps

`kaggle_smoke.ipynb` runs the full 5-stage ZEST pipeline end-to-end on a **tiny ESD subset** and
produces at least one emotion-converted `.wav`. It reuses the shipped `f0.pickle` and the HuBERT-token
manifests (`train_esd.txt` etc.), patches every hardcoded path at runtime, and fixes the
`pickle5` / `torch.load` portability bugs. Nothing in the original repo needs hand-editing.

> This is a **smoke test** (2 epochs, ~200 HiFi-GAN steps) to prove the pipeline runs — not a
> quality run. To reproduce paper-quality audio, raise the knobs in Cell 1 (see bottom).

---

## 1. Get a GPU on Kaggle
1. Sign in at kaggle.com and **verify your phone number** (Settings → Phone) — required for GPU + internet.
2. **Create → New Notebook**.
3. Right panel → **Notebook options**:
   - **Accelerator** → **GPU T4 x2** (free) or **GPU P100**.
   - **Internet** → **On** (mandatory — HuBERT / wav2vec2 / SpeechBrain download from HuggingFace at runtime).
   - Quota: 30 GPU-hrs/week; a session stops after 9h or 20 min idle.

## 2. Add the input
1. **The ZEST repo** — cloned automatically from GitHub by Cell 1 (`https://github.com/vanshpatil16/zest`).
   The repo must be **public** (or use a token URL `https://USER:TOKEN@github.com/...` for a private repo).
   Internet must be **On** (step 1) for the clone to work.
2. **The ESD dataset** — Add Data → search "Emotional Speech Dataset (ESD)" (or upload your own copy).
   Note its English-wav root, e.g. `/kaggle/input/esd`.

   > The ESD `.wav` basenames (e.g. `0016_000651.wav`) must match the manifest entries — that's how
   > tokens and F0 are looked up. Speakers `0011`–`0020` are the English half of ESD.

## 3. Load the notebook and point it at your inputs
1. **File → Import Notebook** → upload `kaggle_smoke.ipynb` (or paste the cells into a new notebook).
2. In **Cell 1**, the repo URL is already set; edit `ESD_WAV_DIR` to match step 2:
   ```python
   REPO_URL    = "https://github.com/vanshpatil16/zest"   # cloned at runtime (must be public)
   ESD_WAV_DIR = "/kaggle/input/esd"                       # root that contains the ESD English .wav files
   ```

## 4. Run it
Run cells **top to bottom** (Run All works). What each cell does and the success signal:

| Cell | Stage | Success signal |
|------|-------|----------------|
| 1 | Setup, paths, env | `CUDA available: True` and a GPU name printed |
| 2 | Install deps (drops `pickle5`) | `dependencies installed` |
| 3 | Patch hardcoded paths + bugs | `All patches applied.` (raises `PATCH MISS` if a file changed) |
| 4 | Build tiny subset + rewrite manifests | `subset wavs -> train N, val N, test N` all **> 0** |
| 5 | EASE (x-vectors → speaker encoder → embeddings) | `EASE embeddings:` count > 0 |
| 6 | F0 predictor (train → contours → SACE feats) | `f0_contours:` and `wav2vec_feats:` > 0 |
| 7 | Train HiFi-GAN (~200 steps) | a `g_*` checkpoint listed |
| 8 | Convert F0 + HiFi-GAN inference | `CONVERTED WAVS: [...]` and an audio player |
| 9 | Run report | non-zero counts across the board |

---

## Knobs (Cell 1) — raise for a fuller / higher-quality run
| Variable | Smoke default | Meaning |
|----------|---------------|---------|
| `UTTS_PER_BUCKET` | 2 | train wavs per (speaker, emotion) |
| `VAL_UTTS` | 1 | val/test wavs per (speaker, emotion) — raise to 3 if Cell 8 finds no conversion pair |
| `P["EASE_EPOCHS"]` | 3 | speaker-encoder epochs |
| `P["F0_EPOCHS"]` | 2 | F0-predictor epochs (paper uses ~500) |
| `P["HIFIGAN_STEPS"]` | 200 | HiFi-GAN steps (paper uses 100k) |
| `cfg["batch_size"]` (Cell 4) | 4 | lower to 2 on CUDA OOM |

---

## Known failures → fix (triage)
The pipeline is wired and the path patches are verified, but heavy stages only run on the GPU, so
budget for one or two runtime fixes. Most likely:

- **`subset wavs -> ... 0`** in Cell 4 → `ESD_WAV_DIR` is wrong or holds non-English speakers. Point it at the folder whose `.wav` names look like `0016_000651.wav`.
- **`from speechbrain.pretrained import ...` ImportError** → newer SpeechBrain moved it. Add a cell:
  `!pip install -q "speechbrain==0.5.16"` (or change the import to `speechbrain.inference`).
- **CUDA OOM** in Cell 6/7 → lower `cfg["batch_size"]` (Cell 4) and re-run from Cell 4.
- **`No converted wavs`** in Cell 8 → the tiny test split had no valid neutral-source / emotional-reference pair. Set `VAL_UTTS = 3` in Cell 1, re-run Cells 4 and 8.
- **`Code audio mismatch` / collate crash** in Cell 6 → a known latent bug in `custom_collate`
  (`pitch_attention_adv.py`, hubert padded to the F0 length). Paste the traceback back here and I'll patch it.

When a stage fails, re-run **only that cell** after the fix — earlier artifacts persist in `/kaggle/working/zest`.

---

## What the notebook changes vs. the original repo
All edits are applied at runtime to the **working copy** (`/kaggle/working/ZEST`); your input repo is untouched:
- `pickle5` → stdlib `pickle` (8 files); `pickle5` removed from `requirements.txt`.
- `torch.load(..., weights_only=False)` for the full-model checkpoints (torch ≥ 2.6 safe).
- `from config import hparams, f0_stats` → `from config import hparams` (the `f0_stats` import never existed).
- Every hardcoded path (`/folder/to/...`, `/home/soumyad/...`, `/ZEST/code/...`) → environment-variable lookups set in Cell 1.
- `pitch_convert.py` hardcoded `sources` list → dynamic (neutral test wavs actually present).
- `HiFi-GAN/inference.py` F0-path join bug (missing separator) → fixed.
