"""Scores-manifest schema: the Stage A (Kaggle) <-> Stage B (local) contract."""
import json

from eval.esd import EMOTIONS

SCHEMA_VERSION = 1
LANGS = {"en", "zh"}
SPLITS = {"dev", "eval"}

_SPK_KEYS = {"conv_file": str, "enroll_speaker": str, "cosine": float,
             "is_target": bool, "cohort_cosines": list, "language": str,
             "split": str, "setting": str}
_EMO_KEYS = {"conv_file": str, "target_emotion": str, "posterior": dict,
             "language": str, "split": str, "setting": str}
_CER_KEYS = {"conv_file": str, "ref": str, "hyp": str, "language": str,
             "split": str, "setting": str}


class ManifestError(ValueError):
	"""Raised when a scores manifest violates the schema."""


def new_manifest(system, git_commit="", models=None):
	return {"meta": {"schema_version": SCHEMA_VERSION, "system": system,
	                 "git_commit": git_commit, "models": models or {}},
	        "speaker_trials": [], "emotion_records": [], "cer_records": []}


def _check_record(rec, keys, kind, idx):
	if not isinstance(rec, dict):
		raise ManifestError(f"{kind}[{idx}]: expected object, got {type(rec).__name__}")
	for k, t in keys.items():
		if k not in rec:
			raise ManifestError(f"{kind}[{idx}]: missing key {k!r}")
		v = rec[k]
		if t is float:
			if isinstance(v, bool) or not isinstance(v, (int, float)):
				raise ManifestError(f"{kind}[{idx}].{k}: expected number")
		elif not isinstance(v, t):
			raise ManifestError(
				f"{kind}[{idx}].{k}: expected {t.__name__}, got {type(v).__name__}")
	if rec["language"] not in LANGS:
		raise ManifestError(f"{kind}[{idx}].language: {rec['language']!r} not in {sorted(LANGS)}")
	if rec["split"] not in SPLITS:
		raise ManifestError(f"{kind}[{idx}].split: {rec['split']!r} not in {sorted(SPLITS)}")


def validate_manifest(m):
	if not isinstance(m, dict):
		raise ManifestError("manifest must be a dict")
	meta = m.get("meta")
	if not isinstance(meta, dict) or not isinstance(meta.get("system"), str) or not meta["system"]:
		raise ManifestError("meta.system missing or empty")
	for section in ("speaker_trials", "emotion_records", "cer_records"):
		if not isinstance(m.get(section), list):
			raise ManifestError(f"{section} missing or not a list")
	for i, rec in enumerate(m["speaker_trials"]):
		_check_record(rec, _SPK_KEYS, "speaker_trials", i)
		if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
		           for x in rec["cohort_cosines"]):
			raise ManifestError(f"speaker_trials[{i}].cohort_cosines: non-numeric entry")
	for i, rec in enumerate(m["emotion_records"]):
		_check_record(rec, _EMO_KEYS, "emotion_records", i)
		if rec["target_emotion"] not in EMOTIONS:
			raise ManifestError(f"emotion_records[{i}].target_emotion: {rec['target_emotion']!r}")
		post = rec["posterior"]
		if set(post.keys()) != set(EMOTIONS):
			raise ManifestError(f"emotion_records[{i}].posterior: keys must be {EMOTIONS}")
		total = sum(float(v) for v in post.values())
		if abs(total - 1.0) > 1e-3:
			raise ManifestError(f"emotion_records[{i}].posterior: sum {total:.4f} != 1")
	for i, rec in enumerate(m["cer_records"]):
		_check_record(rec, _CER_KEYS, "cer_records", i)


def save_manifest(m, path):
	validate_manifest(m)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(m, f, indent=1)


def load_manifest(path):
	with open(path, encoding="utf-8") as f:
		m = json.load(f)
	validate_manifest(m)
	return m
