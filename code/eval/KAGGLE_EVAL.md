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
