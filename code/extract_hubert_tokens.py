import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import HubertModel
from sklearn.cluster import KMeans
import pickle
import argparse


def extract_hubert_tokens(audio_dir, split_file, output_file, kmeans_model=None, layer=9, k=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()

    with open(split_file) as f:
        lines = f.readlines()

    audio_paths = []
    for line in lines:
        if line.startswith("{"):
            import ast
            d = ast.literal_eval(line.strip())
            audio_paths.append(d["audio"])
        else:
            audio_paths.append(line.strip())

    all_features = []
    results = []

    print(f"Extracting HuBERT layer-{layer} features from {len(audio_paths)} files...")
    for path in tqdm(audio_paths):
        sig, sr = torchaudio.load(path)
        if sr != 16000:
            import resampy
            sig = torch.from_numpy(resampy.resample(sig.numpy(), sr, 16000)).float()
        sig = sig.to(device)

        with torch.no_grad():
            out = model(sig, output_hidden_states=True)
            feat = out.hidden_states[layer].squeeze(0).cpu().numpy()
        all_features.append(feat.mean(axis=0))
        results.append((path, feat))

    all_features = np.stack(all_features)

    if kmeans_model is None:
        print(f"Training k-means (K={k}) on {len(all_features)} utterances...")
        kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10)
        kmeans.fit(all_features)
        kmeans_model = kmeans

    print("Quantizing features to discrete tokens...")
    lines_out = []
    for path, feat in tqdm(results):
        tokens = kmeans_model.predict(feat)
        fname = os.path.basename(path)
        lines_out.append(str({"audio": path, "hubert": " ".join(str(t) for t in tokens)}))

    with open(output_file, "w") as f:
        f.write("\n".join(lines_out))

    print(f"Wrote {len(lines_out)} entries to {output_file}")

    if kmeans_model is not None:
        with open("hubert_kmeans.pkl", "wb") as f:
            pickle.dump(kmeans_model, f)
        print("Saved k-means model to hubert_kmeans.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--kmeans_model", default=None)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    kmeans = None
    if args.kmeans_model and os.path.exists(args.kmeans_model):
        with open(args.kmeans_model, "rb") as f:
            kmeans = pickle.load(f)

    extract_hubert_tokens(args.audio_dir, args.split_file, args.output_file, kmeans, args.layer, args.k)
