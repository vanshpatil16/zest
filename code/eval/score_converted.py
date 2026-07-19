"""Stage A (Kaggle, GPU): score converted wavs into a scores manifest.

Hard-fails on: empty converted dir, missing transcript (unless --skip-cer),
model-load failure, NaN embeddings — never emits a silently-empty manifest.
"""
import argparse
import glob
import os
import subprocess
import sys

import torch
import torchaudio

from eval.esd import (EMOTIONS, emotion_from_utt, language_from_speaker,
                      load_split_basenames, parse_converted_name)
from eval.manifest import new_manifest, save_manifest
from eval.train_emotion_probe import EmotionProbe

SR = 16000
_WHISPER_LANG = {"en": "english", "zh": "chinese"}


def _die(msg):
    print(f"[score_converted] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_wav(path, device):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(dim=0, keepdim=True).to(device)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_enrollments(esd_train_dir, per_spk, ecapa, device):
    """Mean ECAPA embedding per speaker over its lowest-numbered real wavs
    (lowest ESD ids are neutral utterances)."""
    by_spk = {}
    for p in sorted(glob.glob(os.path.join(esd_train_dir, "*.wav"))):
        spk = os.path.basename(p)[:4]
        by_spk.setdefault(spk, []).append(p)
    if not by_spk:
        _die(f"no real ESD wavs found in {esd_train_dir}")
    enroll = {}
    for spk, paths in sorted(by_spk.items()):
        embs = []
        for p in paths[:per_spk]:
            with torch.no_grad():
                e = ecapa.encode_batch(_load_wav(p, device)).squeeze()
            embs.append(e)
        emb = torch.stack(embs).mean(dim=0)
        if torch.isnan(emb).any():
            _die(f"NaN enrollment embedding for speaker {spk}")
        enroll[spk] = torch.nn.functional.normalize(emb, dim=0)
    return enroll


def load_transcripts(tsv_path):
    refs = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                utt, text = line.split("\t", 1)
                refs[utt.replace(".wav", "")] = text.strip()
    if not refs:
        _die(f"transcripts file {tsv_path} is empty")
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted-dir", required=True)
    ap.add_argument("--esd-train-dir", required=True,
                    help="dir of real ESD wavs named SPKR_UTTNO.wav (enrollment)")
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--probe", required=True, help="emotion_probe.pth")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcripts-tsv", default=None,
                    help="TSV utt<TAB>text for CER refs")
    ap.add_argument("--skip-cer", action="store_true",
                    help="explicitly skip CER records (no transcripts available)")
    ap.add_argument("--setting", default="DSDT")
    ap.add_argument("--enroll-per-spk", type=int, default=20)
    args = ap.parse_args(argv)

    conv_paths = sorted(glob.glob(os.path.join(args.converted_dir, "*.wav")))
    if not conv_paths:
        _die(f"no converted wavs in {args.converted_dir}")
    if args.transcripts_tsv is None and not args.skip_cer:
        _die("provide --transcripts-tsv or pass --skip-cer explicitly")
    if args.transcripts_tsv is not None and args.skip_cer:
        _die("pass only one of --transcripts-tsv / --skip-cer, not both")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # older speechbrain
        from speechbrain.pretrained import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device})

    from transformers import (HubertModel, Wav2Vec2FeatureExtractor,
                              WhisperForConditionalGeneration, WhisperProcessor)
    hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960").to(device).eval()
    probe = EmotionProbe()
    probe.load_state_dict(torch.load(args.probe, map_location=device,
                                     weights_only=True))
    probe.to(device).eval()

    whisper_proc = whisper = refs = None
    if not args.skip_cer:
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        whisper = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small").to(device).eval()
        refs = load_transcripts(args.transcripts_tsv)

    dev_set = set(load_split_basenames(args.val_split))
    eval_set = set(load_split_basenames(args.test_split))
    enroll = build_enrollments(args.esd_train_dir, args.enroll_per_spk,
                               ecapa, device)

    m = new_manifest(args.system, git_commit=_git_commit(),
                     models={"spk": "speechbrain/spkrec-ecapa-voxceleb",
                             "ser": f"hubert-base probe ({args.probe})",
                             "asr": "openai/whisper-small"})
    for path in conv_paths:
        info = parse_converted_name(path)
        src_base = info["source_utt"] + ".wav"
        if src_base in dev_set:
            split = "dev"
        elif src_base in eval_set:
            split = "eval"
        else:
            _die(f"source {src_base} not in val or test split — "
                 f"cannot assign dev/eval fold")
        lang = language_from_speaker(info["source_speaker"])
        if info["source_speaker"] not in enroll:
            _die(f"no enrollment for source speaker {info['source_speaker']} "
                 f"(missing from --esd-train-dir)")
        wav = _load_wav(path, device)

        with torch.no_grad():
            e = ecapa.encode_batch(wav).squeeze()
        if torch.isnan(e).any():
            _die(f"NaN embedding for {path}")
        e = torch.nn.functional.normalize(e, dim=0)
        cos = {spk: float(torch.dot(e, emb)) for spk, emb in enroll.items()}
        for spk in sorted(enroll):
            cohort = [cos[o] for o in sorted(enroll)
                      if o not in (spk, info["source_speaker"])]
            m["speaker_trials"].append({
                "conv_file": os.path.basename(path), "enroll_speaker": spk,
                "cosine": cos[spk],
                "is_target": spk == info["source_speaker"],
                "cohort_cosines": cohort, "language": lang,
                "split": split, "setting": args.setting})

        with torch.no_grad():
            iv = hubert_fe(wav.squeeze().cpu().numpy(), sampling_rate=SR,
                           return_tensors="pt").input_values.to(device)
            feats = hubert(iv).last_hidden_state.mean(dim=1)
            post = torch.softmax(probe(feats), dim=-1).squeeze().cpu().numpy()
        m["emotion_records"].append({
            "conv_file": os.path.basename(path),
            "target_emotion": emotion_from_utt(info["target_utt"]),
            "posterior": {e_: float(p) for e_, p in zip(EMOTIONS, post)},
            "language": lang, "split": split, "setting": args.setting})

        if not args.skip_cer:
            if info["source_utt"] not in refs:
                _die(f"missing transcript for {info['source_utt']}")
            with torch.no_grad():
                feats_w = whisper_proc(wav.squeeze().cpu().numpy(),
                                       sampling_rate=SR,
                                       return_tensors="pt").input_features
                forced = whisper_proc.get_decoder_prompt_ids(
                    language=_WHISPER_LANG[lang], task="transcribe")
                ids = whisper.generate(feats_w.to(device),
                                       forced_decoder_ids=forced)
            hyp = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0]
            ref = refs[info["source_utt"]]
            if lang == "en":
                ref = "".join(ch for ch in ref.lower() if ch.isalnum() or ch == " ")
                hyp = "".join(ch for ch in hyp.lower() if ch.isalnum() or ch == " ")
            m["cer_records"].append({
                "conv_file": os.path.basename(path), "ref": ref,
                "hyp": hyp.strip(), "language": lang,
                "split": split, "setting": args.setting})

    save_manifest(m, args.out)
    print(f"[score_converted] wrote {args.out}: "
          f"{len(m['speaker_trials'])} speaker trials, "
          f"{len(m['emotion_records'])} emotion records, "
          f"{len(m['cer_records'])} cer records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
