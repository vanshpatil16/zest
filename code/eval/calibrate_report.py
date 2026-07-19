"""Stage B CLI: scores manifest(s) -> calibrated metrics report.

Usage (repo root):
  PYTHONPATH=code python -m eval.calibrate_report --candidate wavlm.json \
      [--baseline w2v2.json] [--out-json report.json] [--out-md report.md]
"""
import argparse
import json
import sys

from eval.manifest import load_manifest
from eval.report import compare_reports, compute_system_report, render_markdown


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True,
                    help="scores manifest of the system under test")
    ap.add_argument("--baseline", default=None,
                    help="optional baseline manifest for A/B comparison")
    ap.add_argument("--out-json", default="report.json")
    ap.add_argument("--out-md", default="report.md")
    args = ap.parse_args(argv)

    cand = compute_system_report(load_manifest(args.candidate))
    base = delta = None
    if args.baseline:
        base = compute_system_report(load_manifest(args.baseline))
        delta = compare_reports(base, cand)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"candidate": cand, "baseline": base, "delta": delta},
                  f, indent=2)
    md = render_markdown(cand, base, delta)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
