import os
import shutil
import argparse
import random
from tqdm import tqdm

SEED = 1234
random.seed(SEED)

EMOTION_RANGES = [
    (0, 350, "neutral"),
    (351, 700, "angry"),
    (701, 1050, "happy"),
    (1051, 1400, "sad"),
    (1401, 99999, "surprise"),
]

def get_emotion_label(file_id):
    file_id = int(file_id)
    for lo, hi, name in EMOTION_RANGES:
        if lo <= file_id <= hi:
            return name
    return "unknown"

def organize_esd(source_dir, output_dir, val_ratio=0.1, test_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    wav_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".wav") and f.count("_") == 1:
                wav_files.append(os.path.join(root, f))

    print(f"Found {len(wav_files)} WAV files")

    speaker_groups = {}
    for path in wav_files:
        fname = os.path.basename(path)
        speaker = fname[:4]
        speaker_groups.setdefault(speaker, []).append(path)

    train_files, val_files, test_files = [], [], []
    for speaker, files in speaker_groups.items():
        files.sort()
        n = len(files)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        random.shuffle(files)
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    for fname, split_files, split_name in [
        ("train_esd.txt", train_files, "train"),
        ("val_esd.txt", val_files, "val"),
        ("test_esd.txt", test_files, "test"),
    ]:
        with open(os.path.join(output_dir, fname), "w") as f:
            for path in split_files:
                f.write(f"{path}\n")

    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        target_dir = os.path.join(output_dir, split_name)
        for src_path in tqdm(split_files, desc=f"Copying {split_name}"):
            dst_path = os.path.join(target_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print("Done. Placeholders created (HuBERT tokens column needs separate extraction).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Path to raw ESD dataset")
    parser.add_argument("--output_dir", default="/ZEST/ESD", help="Output directory")
    args = parser.parse_args()
    organize_esd(args.source_dir, args.output_dir)
