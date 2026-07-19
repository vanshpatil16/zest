"""Shared synthetic-manifest builder for Stage B tests. No GPU, no audio."""
import numpy as np

from eval.esd import EMOTIONS
from eval.manifest import new_manifest

SPK_EN = [f"{i:04d}" for i in range(11, 16)]   # 5 EN speakers
SPK_ZH = [f"{i:04d}" for i in range(1, 6)]     # 5 ZH speakers
_REF_EN = "the quick brown fox jumps over the lazy dog"
_REF_ZH = "今天天气很好我们出去玩"


def make_synth_manifest(system="synth", seed=0, tar_mu=0.6, acc=0.8):
    """Synthetic scores manifest: tar_mu = target-cosine mean (speaker quality),
    acc = emotion-classifier accuracy (emotion quality)."""
    rng = np.random.default_rng(seed)
    m = new_manifest(system, git_commit="synthetic")
    for lang, spks, ref in (("en", SPK_EN, _REF_EN), ("zh", SPK_ZH, _REF_ZH)):
        for split in ("dev", "eval"):
            for src in spks:
                for r in range(6):
                    conv = f"conv_{src}_{split}_{r}.wav"
                    for enroll in spks:
                        is_t = enroll == src
                        mu = tar_mu if is_t else 0.1
                        m["speaker_trials"].append({
                            "conv_file": conv, "enroll_speaker": enroll,
                            "cosine": float(rng.normal(mu, 0.1)),
                            "is_target": bool(is_t),
                            "cohort_cosines": [float(x) for x in
                                               rng.normal(0.1, 0.1, 20)],
                            "language": lang, "split": split, "setting": "DSDT"})
                    true_e = int(rng.integers(0, 5))
                    pred_e = true_e if rng.random() < acc else \
                        int((true_e + 1 + rng.integers(0, 4)) % 5)
                    post = {e: 0.075 for e in EMOTIONS}
                    post[EMOTIONS[pred_e]] = 0.7
                    m["emotion_records"].append({
                        "conv_file": conv, "target_emotion": EMOTIONS[true_e],
                        "posterior": post, "language": lang,
                        "split": split, "setting": "DSDT"})
                    hyp = ref if rng.random() < 0.7 else ref[:-2]
                    m["cer_records"].append({
                        "conv_file": conv, "ref": ref, "hyp": hyp,
                        "language": lang, "split": split, "setting": "DSDT"})
    return m
