import os
import numpy as np
import pickle
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

try:
    import amfm_decompy.basic_tools as basic
    import amfm_decompy.pYAAPT as pYAAPT
except ImportError:
    raise ImportError("amfm_decompy is required. Install via: pip install amfm_decompy")


def extract_yaapt_f0(audio_path, rate=16000, interp=False):
    import soundfile as sf
    y, sr = sf.read(audio_path, dtype="float64")
    if sr != rate:
        import resampy
        y = resampy.resample(y, sr, rate).astype(np.float64)

    frame_length = 25.0
    to_pad = int(frame_length / 1000 * rate) // 2
    y_pad = np.pad(y, (to_pad, to_pad), "constant", constant_values=0)
    signal = basic.SignalObj(y_pad, rate)
    pitch = pYAAPT.yaapt(signal, **{
        'frame_length': frame_length,
        'frame_space': 20.0,
        'nccf_thresh1': 0.25,
        'tda_frame_length': 25.0,
    })
    return pitch.samp_interp if interp else pitch.samp_values


def build_f0_pickle_and_stats(audio_list, output_f0_pickle, output_stats_pth, interp=False):
    f0_dict = {}
    f0_all = []

    print(f"Extracting F0 from {len(audio_list)} files...")
    for audio_path in tqdm(audio_list):
        fname = os.path.basename(audio_path)
        f0 = extract_yaapt_f0(audio_path, interp=interp)
        f0_dict[fname] = f0
        f0_valid = f0[f0 > 0]
        if len(f0_valid) > 0:
            f0_all.extend(f0_valid.tolist())

    with open(output_f0_pickle, "wb") as f:
        pickle.dump(f0_dict, f)
    print(f"Saved F0 pickle ({len(f0_dict)} entries) to {output_f0_pickle}")

    f0_all = np.array(f0_all)
    mean, std = float(np.mean(f0_all)), float(np.std(f0_all))
    stats = {"mean": mean, "std": std}
    torch.save(stats, output_stats_pth)
    print(f"Saved F0 stats (mean={mean:.2f}, std={std:.2f}) to {output_stats_pth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Directory with WAV files")
    parser.add_argument("--output_pickle", default="f0.pickle")
    parser.add_argument("--output_stats", default="esd_f0_stats.pth")
    args = parser.parse_args()

    wav_files = sorted(Path(args.audio_dir).rglob("*.wav"))
    audio_paths = [str(p) for p in wav_files]
    print(f"Found {len(audio_paths)} WAV files in {args.audio_dir}")

    build_f0_pickle_and_stats(audio_paths, args.output_pickle, args.output_stats)
