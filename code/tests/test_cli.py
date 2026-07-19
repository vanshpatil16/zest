import json
from tests.conftest import make_synth_manifest
from eval.manifest import save_manifest
from eval.calibrate_report import main


def test_single_system_run(tmp_path):
    cand = tmp_path / "cand.json"
    save_manifest(make_synth_manifest("wavlm", seed=8), str(cand))
    oj, om = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(["--candidate", str(cand),
               "--out-json", str(oj), "--out-md", str(om)])
    assert rc == 0
    out = json.loads(oj.read_text(encoding="utf-8"))
    assert out["candidate"]["system"] == "wavlm"
    assert out["baseline"] is None and out["delta"] is None
    assert "# ZEST evaluation report" in om.read_text(encoding="utf-8")


def test_ab_run_produces_delta(tmp_path):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    save_manifest(make_synth_manifest("w2v2", seed=9, tar_mu=0.4), str(base))
    save_manifest(make_synth_manifest("wavlm", seed=9, tar_mu=0.7), str(cand))
    oj, om = tmp_path / "r.json", tmp_path / "r.md"
    rc = main(["--candidate", str(cand), "--baseline", str(base),
               "--out-json", str(oj), "--out-md", str(om)])
    assert rc == 0
    out = json.loads(oj.read_text(encoding="utf-8"))
    assert out["delta"]["speaker"]["pooled"]["min_cllr"] < 0
    assert "A/B" in om.read_text(encoding="utf-8")


def test_cli_survives_cp1252_stdout(tmp_path):
    import os
    import subprocess
    import sys as _sys
    from pathlib import Path
    cand = tmp_path / "cand.json"
    save_manifest(make_synth_manifest("wavlm", seed=8), str(cand))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = "code"
    proc = subprocess.run(
        [_sys.executable, "-m", "eval.calibrate_report",
         "--candidate", str(cand),
         "--out-json", str(tmp_path / "r.json"),
         "--out-md", str(tmp_path / "r.md")],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]), env=env)
    assert proc.returncode == 0, proc.stderr
