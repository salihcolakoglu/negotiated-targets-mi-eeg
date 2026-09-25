#!/usr/bin/env python3
"""Entry points for the motor-imagery EEG target experiments."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def run(script, args):
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.call([sys.executable, str(ROOT / script), *map(str, args)], env=env, cwd=ROOT)


def verify(report):
    if report is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report = ROOT / "outputs" / f"verification_{stamp}.json"
    report = Path(report).resolve()
    if report.exists():
        raise FileExistsError(f"Choose a new report path: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eeg_reference_") as tmp:
        base = Path(tmp).resolve()
        with zipfile.ZipFile(ROOT / "reference" / "stored_results.zip") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive entries")
            for item in archive.infolist():
                target = (base / item.filename).resolve()
                if not target.is_relative_to(base) or item.filename.startswith(("/", "\\")):
                    raise ValueError("Invalid archive path")
            archive.extractall(base)
        code = run("src/verify_results.py", ["--data-dir", base, "--report", report, "--quiet"])
    if code:
        return code
    result = json.loads(report.read_text(encoding="utf-8"))
    print(f"PASS: {result['training_runs']} runs; {result['independent_participants']} participants; "
          f"{result['prediction_files_metrics_verified']} saved prediction files checked.")
    print(f"Report: {report}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="Check the stored results; no training or EEG download")
    check.add_argument("--report", type=Path)
    specs = {
        "prepare-csp": ("src/csp/prepare.py", ["--work-dir", "work", "--data-dir", "data"]),
        "train-csp": ("src/csp/train.py", ["--work-dir", "work", "--out", "outputs/csp"]),
        "classical": ("src/csp/classical.py", ["--work-dir", "work", "--out", "outputs/classical"]),
        "prepare-eegnet": ("src/eegnet/prepare.py", ["--reference-raw", "work/Pickle_All_Datasets/Raw_Datasets",
                          "--data-dir", "work/eegnet", "--cache-dir", "data"]),
        "train-eegnet": ("src/eegnet/train.py", ["--data-dir", "work/eegnet", "--out", "outputs/eegnet",
                        "--workers", "2", "--threads", "1"]),
        "summarize": ("src/summarize.py", ["--e1", "outputs/csp/deney01_kosular.jsonl",
                      "--e2", "outputs/eegnet/kosular.jsonl", "--classical", "outputs/classical/classical_results.json",
                      "--out", "outputs/tables"]),
    }
    for name in specs:
        sub.add_parser(name, add_help=False, help="Run " + name.replace("-", " ") + " (use --help for options)")
    args, extra = parser.parse_known_args()
    if args.command == "verify":
        if extra:
            parser.error("Unrecognized arguments: " + " ".join(extra))
        return verify(args.report)
    script, defaults = specs[args.command]
    return run(script, defaults + extra)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
