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
