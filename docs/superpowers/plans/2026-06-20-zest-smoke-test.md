# ZEST Smoke-Test Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing ZEST pipeline (`C:\ZEST\code`) execute end-to-end on a Kaggle GPU notebook against a tiny ESD-English subset, producing at least one converted DSDT `.wav`.

**Architecture:** Overlay a single env-var-backed config module (`zest_paths.py`), patch every hardcoded path and the training-length knobs in the existing stage scripts to read from it, fix portability bugs (`pickle5`, torch≥2.6 `torch.load`), generate a tiny subset + corrected manifests reusing the shipped `f0.pickle`/token files, and drive all five stages from a re-runnable Kaggle notebook.

**Tech Stack:** Python 3.10+, PyTorch, torchaudio, HuggingFace Transformers (`wav2vec2-large-robust-ft-swbd-300h`), SpeechBrain (`spkrec-ecapa-voxceleb`), HiFi-GAN, Kaggle notebooks. Local machine has NO GPU — heavy stages are validated on Kaggle; local checks are syntax-compile + pure-Python unit tests.

**Reference spec:** `docs/superpowers/specs/2026-06-20-zest-smoke-test-reproduction-design.md`

---

## Conventions used by this plan

- All edits are in `C:\ZEST`. Paths below are relative to that root.
- "Local check" = runs on the Windows dev box (no GPU, deps may be absent) → use `py_compile` for syntax and `pytest` for pure-Python logic.
- "Kaggle check" = runs in the notebook (Task 10) on GPU → the real integration test.
- Every patched script gains `import zest_paths as Z` and the notebook sets `PYTHONPATH=<repo>/code` so the import resolves from every stage subdirectory.

## File structure

**New files**
- `code/zest_paths.py` — central config (env-var backed; paths + smoke knobs).
- `code/prepare_subset.py` — Stage 0: select subset from shipped manifests, copy wavs, write corrected subset manifests + HiFi-GAN config json.
- `code/tests/test_zest_paths.py` — local unit test.
- `code/tests/test_prepare_subset.py` — local unit test (synthetic manifest, no ESD/GPU).
- `kaggle_smoke.ipynb` — driver notebook (one cell per stage).
- `code/SMOKE_README.md` — how to run on Kaggle + run-report template.

**Modified files**
- `code/EASE/get_speaker_embedding.py`, `code/EASE/speaker_classifier.py`
- `code/F0_predictor/config.py`, `pitch_attention_adv.py`, `pitch_inference.py`, `get_wav2vec_feats.py`, `pitch_convert.py`
- `code/HiFi-GAN/dataset.py`, `code/HiFi-GAN/inference.py`

---

### Task 1: Initialize git so commits are tracked

**Files:** none (repo metadata only)

- [ ] **Step 1: Init repo and ignore heavy artifacts**

Run (PowerShell, from `C:\ZEST`):
```powershell
git init
@"
__pycache__/
*.pyc
.DS_Store
/data/
/logs/
/kaggle_working/
samples.zip
"@ | Out-File -Encoding utf8 .gitignore
```

- [ ] **Step 2: First commit of current state**

```bash
git add -A
git commit -m "chore: snapshot ZEST repo before smoke-test patches"
```
Expected: a commit is created. (If git is unavailable, skip this task; later commit steps become optional.)

---

### Task 2: Central config module `zest_paths.py`

**Files:**
- Create: `code/zest_paths.py`
- Test: `code/tests/test_zest_paths.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_zest_paths.py`:
```python
import os
import importlib


def test_defaults_and_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEST_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("ZEST_F0_EPOCHS", "7")
    import zest_paths
    importlib.reload(zest_paths)
    # env override is honored
    assert zest_paths.F0_EPOCHS == 7
    # derived dirs live under WORK_DIR
    assert str(zest_paths.EASE_EMB_DIR).startswith(str(tmp_path / "work"))
    # ensure_dirs creates everything
    zest_paths.ensure_dirs()
    assert zest_paths.EASE_EMB_DIR.is_dir()
    assert zest_paths.CKPT_DIR.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest code/tests/test_zest_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zest_paths'`
(Run pytest with `PYTHONPATH=code`: on PowerShell `$env:PYTHONPATH="code"; python -m pytest code/tests/test_zest_paths.py -v`.)

- [ ] **Step 3: Write `code/zest_paths.py`**

```python
"""Central, env-var-backed config for the ZEST smoke-test run.

Every stage script imports this so paths and smoke-size knobs live in ONE place.
Defaults target a Kaggle notebook; override any value via environment variables.
"""
import os
from pathlib import Path


def _s(env, default):
    return os.environ.get(env, default)


def _i(env, default):
    return int(os.environ.get(env, str(default)))


# ---- roots -----------------------------------------------------------------
ZEST_CODE = Path(_s("ZEST_CODE", str(Path(__file__).resolve().parent)))
ESD_WAV_DIR = Path(_s("ESD_WAV_DIR", "/kaggle/input/esd"))          # raw ESD English wavs (searched recursively)
WORK_DIR = Path(_s("ZEST_WORK", "/kaggle/working/zest"))

# ---- working subdirs -------------------------------------------------------
DATA_DIR = WORK_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
XVECTOR_DIR = WORK_DIR / "x_vectors"
EASE_EMB_DIR = WORK_DIR / "EASE_embeddings"
F0_CONTOUR_DIR = WORK_DIR / "f0_contours"
WAV2VEC_FEATS_DIR = WORK_DIR / "wav2vec_feats"
PRED_DSDT_DIR = WORK_DIR / "pred_DSDT_f0"
CKPT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "converted"

# ---- shipped artifacts (reused as-is) --------------------------------------
F0_PICKLE = Path(_s("ZEST_F0_PICKLE", str(ZEST_CODE / "f0.pickle")))
F0_STATS = Path(_s("ZEST_F0_STATS", str(ZEST_CODE / "esd_f0_stats.pth")))
TRAIN_FULL = ZEST_CODE / "train_esd.txt"
VAL_FULL = ZEST_CODE / "val_esd.txt"
TEST_FULL = ZEST_CODE / "test_esd.txt"

# ---- subset manifests (written by prepare_subset.py) -----------------------
TRAIN_MANIFEST = WORK_DIR / "train_esd_subset.txt"
VAL_MANIFEST = WORK_DIR / "val_esd_subset.txt"
TEST_MANIFEST = WORK_DIR / "test_esd_subset.txt"
HIFIGAN_CONFIG = WORK_DIR / "hifigan_kaggle.json"

# ---- smoke-size knobs ------------------------------------------------------
UTTS_PER_SPEAKER = _i("ZEST_UTTS_PER_SPK", 2)   # train utts per emotion per speaker
VAL_UTTS = _i("ZEST_VAL_UTTS", 1)               # val/test utts per emotion per speaker
EASE_EPOCHS = _i("ZEST_EASE_EPOCHS", 3)
F0_EPOCHS = _i("ZEST_F0_EPOCHS", 2)
HIFIGAN_STEPS = _i("ZEST_HIFIGAN_STEPS", 100)
F0_BATCH = _i("ZEST_F0_BATCH", 4)
HIFIGAN_BATCH = _i("ZEST_HIFIGAN_BATCH", 4)

# trailing-slash string forms for scripts that do `folder + name` concatenation
EASE_EMB_DIR_S = str(EASE_EMB_DIR) + os.sep
F0_CONTOUR_DIR_S = str(F0_CONTOUR_DIR) + os.sep
WAV2VEC_FEATS_DIR_S = str(WAV2VEC_FEATS_DIR) + os.sep
PRED_DSDT_DIR_S = str(PRED_DSDT_DIR) + os.sep


def ensure_dirs():
    for d in (DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, XVECTOR_DIR, EASE_EMB_DIR,
              F0_CONTOUR_DIR, WAV2VEC_FEATS_DIR, PRED_DSDT_DIR, CKPT_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_zest_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/zest_paths.py code/tests/test_zest_paths.py
git commit -m "feat: add zest_paths central config for smoke test"
```

---

### Task 3: Subset + manifest builder `prepare_subset.py`

This reuses the shipped manifests (which already carry HuBERT tokens) and `f0.pickle`. It (a) picks a few utterances per emotion per speaker, (b) copies those wavs from the ESD source tree into the working `train/val/test` folders, (c) writes subset manifests with the `audio` path rewritten to the copied location, and (d) writes the HiFi-GAN config json.

**Files:**
- Create: `code/prepare_subset.py`
- Test: `code/tests/test_prepare_subset.py`

- [ ] **Step 1: Write the failing test (pure-Python logic, no ESD/GPU)**

Create `code/tests/test_prepare_subset.py`:
```python
import importlib


def test_emotion_bucket_ranges():
    import prepare_subset
    assert prepare_subset.emotion_bucket(1) == 0       # neutral
    assert prepare_subset.emotion_bucket(350) == 0
    assert prepare_subset.emotion_bucket(351) == 1     # angry
    assert prepare_subset.emotion_bucket(700) == 1
    assert prepare_subset.emotion_bucket(1051) == 3    # sad
    assert prepare_subset.emotion_bucket(1500) == 4    # surprise


def test_select_subset_balances_speakers_and_emotions():
    import prepare_subset
    # synthetic manifest records: 2 speakers, all 5 emotion buckets, 3 utts each
    records = []
    for spk in ("0011", "0012"):
        for base_id in (1, 351, 701, 1051, 1401):
            for k in range(3):
                fid = base_id + k
                name = f"{spk}_{fid:06d}.wav"
                records.append({"audio": f"/orig/{name}", "hubert": "1 2 3", "duration": 0.5})
    chosen = prepare_subset.select_subset(records, utts_per_bucket=2)
    # 2 speakers * 5 buckets * 2 utts = 20
    assert len(chosen) == 20
    from collections import Counter
    c = Counter((r["audio"].split("/")[-1][:4],
                 prepare_subset.emotion_bucket(int(r["audio"].split("/")[-1][5:11]))) for r in chosen)
    assert all(v == 2 for v in c.values())


def test_rewrite_audio_path():
    import prepare_subset
    rec = {"audio": "/home/soumyad/emoconv/ESD/train/0016_000651.wav", "hubert": "5 6", "duration": 1.0}
    out = prepare_subset.rewrite_record(rec, "/work/data/train")
    assert out["audio"] == "/work/data/train/0016_000651.wav"
    assert out["hubert"] == "5 6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_prepare_subset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_subset'`

- [ ] **Step 3: Write `code/prepare_subset.py`**

```python
"""Stage 0: build a tiny ESD-English subset and corrected manifests for the smoke test.

Reuses the shipped *_esd.txt (HuBERT tokens) and f0.pickle. Only the audio FILES
must come from the ESD source tree; everything else is shipped.
"""
import ast
import json
import shutil
from collections import defaultdict
from pathlib import Path

import zest_paths as Z

EMOTION_BOUNDS = [(0, 350), (351, 700), (701, 1050), (1051, 1400), (1401, 10 ** 9)]


def emotion_bucket(file_id: int) -> int:
    for i, (lo, hi) in enumerate(EMOTION_BOUNDS):
        if lo <= file_id <= hi:
            return i
    return 4


def read_manifest(path: Path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def basename(rec) -> str:
    p = rec["audio"]
    return p.split("/")[-1].split("\\")[-1]


def select_subset(records, utts_per_bucket: int):
    """Pick up to `utts_per_bucket` records per (speaker, emotion bucket)."""
    grouped = defaultdict(list)
    for rec in records:
        bn = basename(rec)
        spk = bn[:4]
        fid = int(bn[5:11])
        grouped[(spk, emotion_bucket(fid))].append(rec)
    chosen = []
    for key in sorted(grouped):
        recs = sorted(grouped[key], key=basename)
        chosen.extend(recs[:utts_per_bucket])
    return chosen


def rewrite_record(rec, dest_dir: str):
    out = dict(rec)
    out["audio"] = str(Path(dest_dir) / basename(rec)).replace("\\", "/")
    return out


def index_source_wavs(src_root: Path):
    """basename -> absolute path, for every wav under the ESD source tree."""
    idx = {}
    for p in Path(src_root).rglob("*.wav"):
        idx.setdefault(p.name, str(p))
    return idx


def build_split(full_manifest: Path, dest_dir: Path, manifest_out: Path,
                utts_per_bucket: int, wav_index):
    records = read_manifest(full_manifest)
    chosen = select_subset(records, utts_per_bucket)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for rec in chosen:
        bn = basename(rec)
        src = wav_index.get(bn)
        if src is None:
            missing.append(bn)
            continue
        dst = dest_dir / bn
        if not dst.exists():
            shutil.copy2(src, dst)
        written.append(rewrite_record(rec, str(dest_dir)))
    with open(manifest_out, "w") as f:
        f.write("\n".join(str(r) for r in written))
    return len(written), missing


def write_hifigan_config():
    """Render a Kaggle HiFi-GAN config from the shipped template values."""
    cfg = {
        "input_training_file": str(Z.TRAIN_MANIFEST),
        "input_validation_file": str(Z.VAL_MANIFEST),
        "resblock": "1", "num_gpus": 0, "batch_size": Z.HIFIGAN_BATCH,
        "learning_rate": 0.0002, "adam_b1": 0.8, "adam_b2": 0.99,
        "lr_decay": 0.999, "seed": 1234,
        "upsample_rates": [5, 4, 4, 2, 2], "upsample_kernel_sizes": [11, 8, 8, 4, 4],
        "upsample_initial_channel": 512, "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "num_embeddings": 100, "embedding_dim": 128, "model_in_dim": 512,
        "segment_size": 8960, "code_hop_size": 320, "f0": True, "multispkr": "_",
        "encodeunits": "_", "encodef0": "_", "num_mels": 80, "num_freq": 1025,
        "n_fft": 1024, "hop_size": 256, "win_size": 1024,
        "f0_stats": str(Z.F0_STATS), "f0_normalize": True, "f0_feats": False,
        "f0_median": False, "f0_interp": False, "sampling_rate": 16000,
        "fmin": 0, "fmax": 8000, "fmax_for_loss": None, "num_workers": 0,
        "dist_config": {"dist_backend": "nccl", "dist_url": "env://"},
    }
    with open(Z.HIFIGAN_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    Z.ensure_dirs()
    wav_index = index_source_wavs(Z.ESD_WAV_DIR)
    print(f"Indexed {len(wav_index)} source wavs under {Z.ESD_WAV_DIR}")
    for full, dest, out, n in (
        (Z.TRAIN_FULL, Z.TRAIN_DIR, Z.TRAIN_MANIFEST, Z.UTTS_PER_SPEAKER),
        (Z.VAL_FULL, Z.VAL_DIR, Z.VAL_MANIFEST, Z.VAL_UTTS),
        (Z.TEST_FULL, Z.TEST_DIR, Z.TEST_MANIFEST, Z.VAL_UTTS),
    ):
        count, missing = build_split(full, dest, out, n, wav_index)
        print(f"{out.name}: wrote {count} files; missing {len(missing)}")
        if missing:
            print("  e.g. missing:", missing[:5])
    write_hifigan_config()
    print(f"HiFi-GAN config -> {Z.HIFIGAN_CONFIG}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="code"; python -m pytest code/tests/test_prepare_subset.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add code/prepare_subset.py code/tests/test_prepare_subset.py
git commit -m "feat: add prepare_subset stage-0 builder"
```

---

### Task 4: Portability fixes across the repo

Replace `import pickle5 as pickle` with stdlib `pickle`, and make full-object `torch.load` calls torch≥2.6-safe (`weights_only=False`). Also fix the latent `f0_stats` ImportError in `get_wav2vec_feats.py`.

**Files:**
- Modify: `code/F0_predictor/pitch_attention_adv.py:19`, `code/F0_predictor/pitch_inference.py:18`, `code/F0_predictor/get_wav2vec_feats.py:17,20`, `code/F0_predictor/pitch_convert.py:18`, `code/EASE/speaker_classifier.py:15`, `code/HiFi-GAN/dataset.py:22`

- [ ] **Step 1: Swap pickle5 → pickle (all six files)**

In each file replace the line `import pickle5 as pickle` with:
```python
import pickle  # was: pickle5 (unavailable on modern Python; stdlib reads protocol 5)
```

- [ ] **Step 2: Fix `get_wav2vec_feats.py:17` ImportError**

`from config import hparams, f0_stats` → (config.py has no `f0_stats`)
```python
from config import hparams
```

- [ ] **Step 3: Make full-model loads torch≥2.6-safe**

In `pitch_inference.py:182`, `get_wav2vec_feats.py:174`, `pitch_convert.py:176` replace
`model = torch.load('f0_predictor.pth', map_location=device)` with:
```python
model = torch.load('f0_predictor.pth', map_location=device, weights_only=False)
```
In `EASE/speaker_classifier.py:210` replace
`model = torch.load('EASE.pth', map_location=device)` with:
```python
model = torch.load('EASE.pth', map_location=device, weights_only=False)
```

- [ ] **Step 4: Local syntax check**

Run: `python -m py_compile code/F0_predictor/pitch_attention_adv.py code/F0_predictor/pitch_inference.py code/F0_predictor/get_wav2vec_feats.py code/F0_predictor/pitch_convert.py code/EASE/speaker_classifier.py code/HiFi-GAN/dataset.py`
Expected: no output (exit 0).

- [ ] **Step 5: Commit**

```bash
git add code/F0_predictor code/EASE code/HiFi-GAN
git commit -m "fix: portability (stdlib pickle, weights_only=False, f0_stats import)"
```

---

### Task 5: Patch the EASE stage to use `zest_paths`

**Files:**
- Modify: `code/EASE/get_speaker_embedding.py:8-9`
- Modify: `code/EASE/speaker_classifier.py:119-134,147,213,221,229,237`

- [ ] **Step 1: `get_speaker_embedding.py` — paths from config**

Replace lines 8–9:
```python
folder = "/folder/to/wav_files"
target_folder = "/folder/to/store/x-vectors"
```
with (also add the two imports near the top):
```python
import sys
import zest_paths as Z
# wav source folder is passed as argv[1] (train/val/test); x-vectors go to one shared dir
folder = sys.argv[1] if len(sys.argv) > 1 else str(Z.TRAIN_DIR)
target_folder = str(Z.XVECTOR_DIR)
```

- [ ] **Step 2: `speaker_classifier.py` — `create_dataset` paths**

Replace the `create_dataset` head (lines 119–127):
```python
def create_dataset(mode, bs=32):
    speaker_folder = "/folder/to/x-vectors"
    if mode == 'train':
        folder = "/folder/to/train/audio/files"
    elif mode == 'val':
        folder = "/folder/to/validation/audio/files"
    elif mode =="test":
        folder = "/folder/to/test/audio/files"
```
with:
```python
def create_dataset(mode, bs=32):
    import zest_paths as Z
    speaker_folder = str(Z.XVECTOR_DIR)
    if mode == 'train':
        folder = str(Z.TRAIN_DIR)
    elif mode == 'val':
        folder = str(Z.VAL_DIR)
    elif mode == "test":
        folder = str(Z.TEST_DIR)
```

- [ ] **Step 3: `speaker_classifier.py` — epochs + embeddings dir**

Replace the epochs header at line 147 `    for e in range(10):` with:
```python
    import zest_paths as Z
    for e in range(Z.EASE_EPOCHS):
```
In `get_embedding()` replace line 213 `os.makedirs("EASE_embeddings", exist_ok=True)` with:
```python
    import zest_paths as Z
    os.makedirs(str(Z.EASE_EMB_DIR), exist_ok=True)
```
and in the three `np.save(os.path.join("EASE_embeddings", target_file_name), ...)` sites (lines 221, 229, 237) replace the literal `"EASE_embeddings"` with `str(Z.EASE_EMB_DIR)`.

- [ ] **Step 4: Local syntax check**

Run: `python -m py_compile code/EASE/get_speaker_embedding.py code/EASE/speaker_classifier.py`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add code/EASE
git commit -m "feat: route EASE stage through zest_paths"
```

---

### Task 6: Patch F0_predictor `config.py` + `pitch_attention_adv.py`

**Files:**
- Modify: `code/F0_predictor/config.py:1-10`
- Modify: `code/F0_predictor/pitch_attention_adv.py:17(imports),84,265-274,308`

- [ ] **Step 1: `config.py` — paths from `zest_paths`**

Replace lines 1–10:
```python
train_datasets = {"ESD":"/home/soumyad/emoconv/ESD/train"}
val_datasets = {"ESD":"/home/soumyad/emoconv/ESD/val"}
test_datasets = {"ESD":"/home/soumyad/emoconv/ESD/test"}


train_tokens_orig = {"ESD":"/ZEST/code/train_esd.txt"}
val_tokens_orig = {"ESD":"/ZEST/code/val_esd.txt"}
test_tokens_orig = {"ESD":"/ZEST/code/test_esd.txt"}

f0_file = "ZEST/code/f0.pickle"
```
with:
```python
import zest_paths as Z

train_datasets = {"ESD": str(Z.TRAIN_DIR)}
val_datasets = {"ESD": str(Z.VAL_DIR)}
test_datasets = {"ESD": str(Z.TEST_DIR)}

train_tokens_orig = {"ESD": str(Z.TRAIN_MANIFEST)}
val_tokens_orig = {"ESD": str(Z.VAL_MANIFEST)}
test_tokens_orig = {"ESD": str(Z.TEST_MANIFEST)}

f0_file = str(Z.F0_PICKLE)
```

- [ ] **Step 2: `pitch_attention_adv.py` — add import + EASE path (line 84)**

Add `import zest_paths as Z` to the import block (e.g. after line 17 `from config import hparams`). Then replace line 84:
```python
        speaker_feature = np.load(os.path.join("/folder/to/EASE/embeddings", file_name.replace(".wav", ".npy")))
```
with:
```python
        speaker_feature = np.load(os.path.join(str(Z.EASE_EMB_DIR), file_name.replace(".wav", ".npy")))
```

- [ ] **Step 3: `pitch_attention_adv.py` — dataset folders + batch (lines 265–274)**

Replace the `create_dataset` head:
```python
def create_dataset(mode, bs=24):
    if mode == 'train':
        folder = "/folder/to/train/audio/files"
        token_file = train_tokens_orig["ESD"]
    elif mode == 'val':
        folder = "/folder/to/validation/audio/files"
        token_file = val_tokens_orig["ESD"]
    else:
        folder = "/folder/to/test/audio/files"
        token_file = test_tokens_orig["ESD"]
```
with:
```python
def create_dataset(mode, bs=None):
    if bs is None:
        bs = Z.F0_BATCH
    if mode == 'train':
        folder = str(Z.TRAIN_DIR)
        token_file = train_tokens_orig["ESD"]
    elif mode == 'val':
        folder = str(Z.VAL_DIR)
        token_file = val_tokens_orig["ESD"]
    else:
        folder = str(Z.TEST_DIR)
        token_file = test_tokens_orig["ESD"]
```

- [ ] **Step 4: `pitch_attention_adv.py` — epochs (line 308)**

Replace `    for e in range(500):` with:
```python
    for e in range(Z.F0_EPOCHS):
```

- [ ] **Step 5: Local syntax check**

Run: `python -m py_compile code/F0_predictor/config.py code/F0_predictor/pitch_attention_adv.py`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add code/F0_predictor/config.py code/F0_predictor/pitch_attention_adv.py
git commit -m "feat: route F0 predictor training through zest_paths"
```

---

### Task 7: Patch F0_predictor inference/conversion scripts

`pitch_inference.py` and `get_wav2vec_feats.py` write into hardcoded `"f0_contours"` / `"wav2vec_feats"`; redirect to `zest_paths`. `pitch_convert.py` has a hardcoded source list and `"pred_DSDT_f0"` output; make the source list dynamic and redirect output.

**Files:**
- Modify: `code/F0_predictor/pitch_inference.py:179,199,214,229`
- Modify: `code/F0_predictor/get_wav2vec_feats.py:169-170`
- Modify: `code/F0_predictor/pitch_convert.py:173,179-182,214`

- [ ] **Step 1: `pitch_inference.py` — output dir**

Add `import zest_paths as Z` to imports. Replace line 179 `    os.makedirs("f0_contours", exist_ok=True)` with:
```python
    os.makedirs(str(Z.F0_CONTOUR_DIR), exist_ok=True)
```
In each of the three `np.save(os.path.join("f0_contours", target_file_name), ...)` (lines 199, 214, 229) replace the literal `"f0_contours"` with `str(Z.F0_CONTOUR_DIR)`.

- [ ] **Step 2: `get_wav2vec_feats.py` — output dir**

Add `import zest_paths as Z`. Replace lines 169–170:
```python
    wav2vec_feats_folder = "wav2vec_feats"
    os.makedirs(wav2vec_feats_folder, exist_ok=True)
```
with:
```python
    wav2vec_feats_folder = str(Z.WAV2VEC_FEATS_DIR)
    os.makedirs(wav2vec_feats_folder, exist_ok=True)
```
(The three `np.save(os.path.join(wav2vec_feats_folder, ...))` sites then resolve correctly with no further change.)

- [ ] **Step 3: `pitch_convert.py` — dynamic sources + output dir**

Add `import zest_paths as Z`. Replace line 173 `    os.makedirs("pred_DSDT_f0", exist_ok=True)` with:
```python
    os.makedirs(str(Z.PRED_DSDT_DIR), exist_ok=True)
```
Replace the hardcoded `sources = [...]` block (lines 179–182) with a dynamic list of neutral (id<350) test wavs actually present:
```python
    sources = sorted(
        f for f in os.listdir(str(Z.TEST_DIR))
        if f.endswith(".wav") and int(f[5:11]) < 350
    )
```
Replace line 214 `np.save(os.path.join("pred_DSDT_f0", final_name), ...)` literal `"pred_DSDT_f0"` with `str(Z.PRED_DSDT_DIR)`.

- [ ] **Step 4: Local syntax check**

Run: `python -m py_compile code/F0_predictor/pitch_inference.py code/F0_predictor/get_wav2vec_feats.py code/F0_predictor/pitch_convert.py`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add code/F0_predictor
git commit -m "feat: route F0 inference/conversion through zest_paths"
```

---

### Task 8: Patch HiFi-GAN dataset (hardcoded EASE path)

**Files:**
- Modify: `code/HiFi-GAN/dataset.py:22(imports),291`

- [ ] **Step 1: EASE embeddings path from config**

Add `import zest_paths as Z` to the imports block (top of file, near line 22). Replace line 291:
```python
            feats['spkr'] = np.load("/ZEST/code/EASE/EASE_embeddings/" + emo_file_name)
```
with:
```python
            feats['spkr'] = np.load(Z.EASE_EMB_DIR_S + emo_file_name)
```

- [ ] **Step 2: Local syntax check**

Run: `python -m py_compile code/HiFi-GAN/dataset.py`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add code/HiFi-GAN/dataset.py
git commit -m "fix: HiFi-GAN dataset reads EASE embeddings from zest_paths"
```

---

### Task 9: Patch HiFi-GAN inference (conversion) + add reconstruction fallback

The conversion path has three hardcoded paths and a join bug; fix them. Also add a tiny `--recon` branch so the smoke test still produces a wav if the conversion reference-matching yields nothing on the small subset.

**Files:**
- Modify: `code/HiFi-GAN/inference.py:31(imports),192,203,205,226 and the inference() function`

- [ ] **Step 1: Add config import**

After line 31 (`from librosa.util import normalize`) add:
```python
import zest_paths as Z
```

- [ ] **Step 2: Fix the three hardcoded conversion paths + join bug (lines 192, 203, 205)**

Replace line 192:
```python
            reference_files = os.listdir("/folder/to/ESD/test/wavs")
```
with:
```python
            reference_files = os.listdir(str(Z.TEST_DIR))
```
Replace line 203:
```python
                emo_embed = np.load("/ZEST/code/F0_predictor/wav2vec_feats/" + filename.replace(".wav", ".npy"))
```
with:
```python
                emo_embed = np.load(Z.WAV2VEC_FEATS_DIR_S + filename.replace(".wav", ".npy"))
```
Replace line 205 (the original is missing a path separator between the dir and the name):
```python
                f0 = np.load("/ZEST/code/F0_predictor/pred_DSDT_f0" + fname_out_name + filename.replace(".wav", ".npy"))
```
with:
```python
                f0 = np.load(Z.PRED_DSDT_DIR_S + fname_out_name + filename.replace(".wav", ".npy"))
```

- [ ] **Step 3: Default `--input_code_file` to the subset manifest + add `--recon` (line 226)**

Replace:
```python
    parser.add_argument('--input_code_file', default="/ZEST/code/test_esd.txt")
```
with:
```python
    parser.add_argument('--input_code_file', default=str(Z.TEST_MANIFEST))
    parser.add_argument('--recon', action='store_true', help='reconstruct the source (smoke fallback, no reference)')
```

- [ ] **Step 4: Add a reconstruction branch in `inference()`**

Inside `inference()`, after the `new_code = dict(code)` block (the `if 'f0' in new_code:` lines, ~line 188) and before `if h.get('multispkr', None) and a.convert:` (line 190), insert:
```python
        if a.recon:
            audio, rtf = generate(h, generator, code)
            output_file = os.path.join(a.output_dir, "recon_" + fname_out_name + ".wav")
            audio = librosa.util.normalize(audio.astype(np.float32))
            write(output_file, h.sampling_rate, audio)
            return
```
(`code` already carries `code`, `f0`, `spkr`, `emo_embed` from `__getitem__`, so reconstruction needs no reference files.)

- [ ] **Step 5: Local syntax check**

Run: `python -m py_compile code/HiFi-GAN/inference.py`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add code/HiFi-GAN/inference.py
git commit -m "feat: HiFi-GAN inference via zest_paths + recon fallback"
```

---

### Task 10: Kaggle driver notebook `kaggle_smoke.ipynb`

Build a notebook with one code cell per stage; each cell sets env vars, runs the stage as a subprocess from the correct directory, and prints artifact counts. Cells are independently re-runnable.

**Files:**
- Create: `kaggle_smoke.ipynb`

- [ ] **Step 1: Author the notebook cells (content spec)**

Create `kaggle_smoke.ipynb` with these code cells, in order:

**Cell 0 — setup & env**
```python
import os, sys, subprocess
REPO = "/kaggle/working/ZEST"            # adjust if the repo is mounted elsewhere
CODE = f"{REPO}/code"
os.environ["ZEST_CODE"] = CODE
os.environ["ESD_WAV_DIR"] = "/kaggle/input/esd"   # <-- point at your ESD English wavs
os.environ["ZEST_WORK"] = "/kaggle/working/zest"
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["ZEST_UTTS_PER_SPK"] = "2"
os.environ["ZEST_EASE_EPOCHS"] = "3"
os.environ["ZEST_F0_EPOCHS"] = "2"
os.environ["ZEST_HIFIGAN_STEPS"] = "100"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO}/requirements.txt"], check=True)
print("Internet must be ENABLED for HF/SpeechBrain downloads. torch:", __import__("torch").__version__)
```

**Cell 1 — Stage 0 data prep**
```python
subprocess.run([sys.executable, f"{CODE}/prepare_subset.py"], cwd=CODE, check=True)
import zest_paths as Z
for d in (Z.TRAIN_DIR, Z.VAL_DIR, Z.TEST_DIR):
    print(d, "->", len(list(d.glob('*.wav'))), "wavs")
```

**Cell 2 — Stage 1 EASE**
```python
import zest_paths as Z
EASE = f"{CODE}/EASE"
for split, folder in (("train", Z.TRAIN_DIR), ("val", Z.VAL_DIR), ("test", Z.TEST_DIR)):
    subprocess.run([sys.executable, f"{EASE}/get_speaker_embedding.py", str(folder)], cwd=EASE, check=True)
subprocess.run([sys.executable, f"{EASE}/speaker_classifier.py"], cwd=EASE, check=True)
print("EASE embeddings:", len(list(Z.EASE_EMB_DIR.glob('*.npy'))))
```

**Cell 3 — Stage 2 F0 predictor (train → contours → SACE feats)**
```python
import zest_paths as Z
F0 = f"{CODE}/F0_predictor"
subprocess.run([sys.executable, f"{F0}/pitch_attention_adv.py"], cwd=F0, check=True)
subprocess.run([sys.executable, f"{F0}/pitch_inference.py"], cwd=F0, check=True)
subprocess.run([sys.executable, f"{F0}/get_wav2vec_feats.py"], cwd=F0, check=True)
print("f0_contours:", len(list(Z.F0_CONTOUR_DIR.glob('*.npy'))),
      "wav2vec_feats:", len(list(Z.WAV2VEC_FEATS_DIR.glob('*.npy'))))
```
Note: `pitch_attention_adv.py` writes `f0_predictor.pth` into its cwd (`F0_predictor/`); the inference scripts load it from cwd — keep `cwd=F0` for all three.

**Cell 4 — Stage 3 HiFi-GAN train**
```python
import zest_paths as Z
HG = f"{CODE}/HiFi-GAN"
subprocess.run([sys.executable, f"{HG}/train.py",
                "--checkpoint_path", str(Z.CKPT_DIR),
                "--config", str(Z.HIFIGAN_CONFIG),
                "--pitch_folder", Z.F0_CONTOUR_DIR_S,
                "--emo_folder", Z.WAV2VEC_FEATS_DIR_S,
                "--training_steps", os.environ["ZEST_HIFIGAN_STEPS"],
                "--checkpoint_interval", os.environ["ZEST_HIFIGAN_STEPS"]], cwd=HG, check=True)
print("checkpoints:", [p.name for p in Z.CKPT_DIR.glob('g_*')])
```

**Cell 5 — Stage 4 conversion (+ recon fallback)**
```python
import zest_paths as Z
F0 = f"{CODE}/F0_predictor"; HG = f"{CODE}/HiFi-GAN"
subprocess.run([sys.executable, f"{F0}/pitch_convert.py"], cwd=F0, check=True)
print("pred_DSDT_f0:", len(list(Z.PRED_DSDT_DIR.glob('*.npy'))))
common = ["--checkpoint_file", str(Z.CKPT_DIR), "--output_dir", str(Z.OUTPUT_DIR),
          "--emo_folder", Z.WAV2VEC_FEATS_DIR_S, "--pitch_folder", Z.F0_CONTOUR_DIR_S,
          "--f0-stats", str(Z.F0_STATS), "--input_code_file", str(Z.TEST_MANIFEST), "--debug"]
subprocess.run([sys.executable, f"{HG}/inference.py", "--convert"] + common, cwd=HG, check=False)
wavs = list(Z.OUTPUT_DIR.glob('*.wav'))
if not wavs:
    print("Conversion produced no wavs; running reconstruction fallback...")
    subprocess.run([sys.executable, f"{HG}/inference.py", "--recon"] + common, cwd=HG, check=True)
    wavs = list(Z.OUTPUT_DIR.glob('*.wav'))
print("OUTPUT WAVS:", [w.name for w in wavs][:10])
import IPython.display as ipd
if wavs: ipd.display(ipd.Audio(str(wavs[0])))
```

**Cell 6 — run report**
```python
import zest_paths as Z
print("=== ZEST smoke-test run report ===")
for label, d, pat in [("subset train", Z.TRAIN_DIR, "*.wav"), ("EASE emb", Z.EASE_EMB_DIR, "*.npy"),
                      ("f0_contours", Z.F0_CONTOUR_DIR, "*.npy"), ("wav2vec_feats", Z.WAV2VEC_FEATS_DIR, "*.npy"),
                      ("checkpoints", Z.CKPT_DIR, "g_*"), ("converted wavs", Z.OUTPUT_DIR, "*.wav")]:
    print(f"{label:16s}: {len(list(d.glob(pat)))}")
```

- [ ] **Step 2: Validate the notebook JSON parses**

Run: `python -c "import json; json.load(open(r'kaggle_smoke.ipynb'))"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add kaggle_smoke.ipynb
git commit -m "feat: add Kaggle smoke-test driver notebook"
```

---

### Task 11: Run README + first Kaggle execution

**Files:**
- Create: `code/SMOKE_README.md`

- [ ] **Step 1: Write `code/SMOKE_README.md`**

Document, concretely: (1) create a Kaggle notebook with GPU + Internet ON; (2) add the repo and the ESD-English dataset as inputs (or upload/clone the repo to `/kaggle/working/ZEST`); (3) set `ESD_WAV_DIR` in Cell 0 to the ESD English wav root; (4) run cells 0→6 top to bottom; (5) success = Cell 6 reports ≥1 converted (or recon) wav and Cell 5 plays audio. Include the env-var knob table (`ZEST_UTTS_PER_SPK`, `ZEST_VAL_UTTS`, `ZEST_EASE_EPOCHS`, `ZEST_F0_EPOCHS`, `ZEST_HIFIGAN_STEPS`, `ZEST_F0_BATCH`, `ZEST_HIFIGAN_BATCH`) and a short "known failure → fix" list (Internet off → enable; `weights_only` error → already patched; empty conversion → recon fallback; `amfm_decompy` build → `pip install amfm_decompy`; CUDA OOM → lower `ZEST_*_BATCH`).

- [ ] **Step 2: Commit**

```bash
git add code/SMOKE_README.md
git commit -m "docs: add Kaggle smoke-test README"
```

- [ ] **Step 3: Kaggle integration run (the real test)**

On Kaggle, run cells 0→6. Expected per-stage:
- Cell 1: `train/val/test` folders each report a non-zero wav count; `missing 0` (if missing > 0, `ESD_WAV_DIR` is wrong).
- Cell 2: `EASE embeddings:` ≥ subset size.
- Cell 3: `f0_contours` and `wav2vec_feats` counts ≈ subset size.
- Cell 4: at least one `g_*` checkpoint written.
- Cell 5: `OUTPUT WAVS` non-empty; audio player renders.
- Cell 6: report shows non-zero across the board.

- [ ] **Step 4: Triage loop**

For each stage failure, paste the traceback back into this session. Fix forward (most likely: device/dtype mismatch in a training loop, the `custom_collate` hubert-padding `max_len_f0` bug at `pitch_attention_adv.py:243` if it crashes, or YAAPT install). Commit each fix with a `fix:` message and re-run only the affected cell.

---

## Self-review

**Spec coverage:**
- Reuse shipped f0.pickle/manifests/stats → Task 3 (no regeneration). ✓
- Central config overlay → Task 2. ✓
- Patch ~10 hardcoded sites → Tasks 5–9 (EASE ×2, F0 config + train, F0 inference ×3, HiFi-GAN dataset, HiFi-GAN inference). ✓
- Portability fixes (pickle5, weights_only) → Task 4 (+ the `f0_stats` ImportError found during audit). ✓
- Stage-0 subset = all 10 speakers × few utts, rewrite manifest audio paths → Task 3. ✓
- Kaggle notebook, one cell per stage → Task 10. ✓
- End state = converted DSDT wav (with recon fallback) → Task 9 + Cell 5. ✓
- Run report → Cell 6 + Task 11. ✓
- Git note (repo not initialized) → Task 1 initializes it so commit steps work. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — every edit shows exact before/after code and exact line anchors. The SMOKE_README (Task 11) is described by required contents, not code, which is appropriate for prose docs.

**Type/name consistency:** `zest_paths` symbol names (`TRAIN_DIR`, `VAL_DIR`, `TEST_DIR`, `XVECTOR_DIR`, `EASE_EMB_DIR`, `EASE_EMB_DIR_S`, `F0_CONTOUR_DIR(_S)`, `WAV2VEC_FEATS_DIR(_S)`, `PRED_DSDT_DIR(_S)`, `CKPT_DIR`, `OUTPUT_DIR`, `TRAIN_MANIFEST`, `VAL_MANIFEST`, `TEST_MANIFEST`, `HIFIGAN_CONFIG`, `F0_PICKLE`, `F0_STATS`, `UTTS_PER_SPEAKER`, `VAL_UTTS`, `F0_EPOCHS`, `EASE_EPOCHS`, `HIFIGAN_STEPS`, `F0_BATCH`, `HIFIGAN_BATCH`, `ensure_dirs`) are defined in Task 2 and used identically in Tasks 3,5,6,7,8,9,10. The trailing-slash `*_S` forms are used exactly where scripts do `folder + name` concatenation (`dataset.py:291`, `inference.py:203,205`); `str(...)` dir forms are used where scripts do `os.path.join`/`os.makedirs`/`os.listdir`. Consistent.

**Known residual risks (carried from spec, handled in Task 11 triage):** torch-version quirks beyond `weights_only`, YAAPT/`amfm_decompy` build, and the `custom_collate` `max_len_f0` latent bug — all surface only at Kaggle runtime and are fixed forward.
