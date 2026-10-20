#!/usr/bin/env python3
"""Final Lee analysis: require all 1,350 verified fits before any test summary.

Each of 54 participants contributes one seed-averaged paired observation.
The five predeclared contrasts form one two-sided paired-test/Holm family.
Incomplete directories produce only completion accounting, never interim tests.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import stats

CONDITIONS = ("S0", "W0", "W1", "W2", "W3")
SEEDS = (1, 2, 3, 4, 5)
SUBJECTS = tuple(range(1, 55))
CONTRASTS = (("W2", "W0"), ("W1", "W0"), ("W2", "W1"), ("W2", "W3"), ("W0", "S0"))
PARAMS = dict(lr=0.001, epochs_per_phase=30, phases=10, batch=16, dropout=0.5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def holm_adjust(pvalues):
    """Step-down Holm adjustment, retaining the original contrast order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm requires finite probabilities")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_summary(first, second):
    """Difference is first minus second; accuracy inputs are fractions."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Paired arrays must have equal one-dimensional shape and n >= 2")
    difference = first - second
    if not np.isfinite(difference).all():
        raise ValueError("Non-finite paired observations")
    n = len(difference)
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        t = None if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        low, high = mean, mean
        dz = None
        special = "constant_nonzero_difference" if mean != 0 else "all_differences_zero"
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        radius = float(stats.t.ppf(0.975, df=n - 1)) * se
        low, high = mean - radius, mean + radius
        dz = mean / sd
        special = None
    return dict(n_participants=n, mean_difference_percentage_points=100 * mean,
                ci95_low_percentage_points=100 * low, ci95_high_percentage_points=100 * high,
                ci_scope="pointwise unadjusted 95% t interval", t_statistic=t,
                degrees_of_freedom=n - 1, p_two_sided=p, paired_dz=dz,
                special_case=special)


def expected_keys():
    return {(s, c, seed) for s in SUBJECTS for c in CONDITIONS for seed in SEEDS}


def audit_complete(results):
    """Inspect completion/identity/integrity only; never summarize test outcomes."""
    results = Path(results).resolve()
    expected = expected_keys()
    found, failures = {}, []
    common_identity = None
    participant_identity = {}
    for path in sorted(results.rglob("records/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            identity = record["identity"]
            key = (identity["subject"], identity["condition"], identity["seed"])
            if key not in expected or key in found:
                raise ValueError("Unexpected or duplicate participant/condition/seed")
            if identity["command"] != "run" or identity["dataset"] != "Lee2019_MI":
                raise ValueError("Only final Lee runs are eligible; timing output is excluded")
            if identity["parameters"] != PARAMS or record["epochs_completed"] != 300:
                raise ValueError("Training budget differs from the fixed protocol")
            if record["phase_test_metrics_computed"] is not False or record["final_test_evaluations"] != 1:
                raise ValueError("Expected final-only test evaluation")
            if record["n_train"] != 100 or record["n_test"] != 100:
                raise ValueError("Unexpected offline-day trial count")
            shared = {k: v for k, v in identity.items() if k not in ("subject", "condition", "seed", "data_sha256")}
            if common_identity is None:
                common_identity = shared
            elif common_identity != shared:
                raise ValueError("Different code, software, device or protocol identities")
            if key[0] in participant_identity and participant_identity[key[0]] != identity["data_sha256"]:
                raise ValueError("Different input files were used for the same participant")
            participant_identity[key[0]] = identity["data_sha256"]
            run_root = path.parent.parent.resolve()
            prediction = (run_root / record["prediction_file"]).resolve()
            if not prediction.is_relative_to(run_root) or not prediction.is_relative_to(results) or not prediction.is_file():
                raise ValueError("Prediction file is absent or outside the result directory")
            if sha(prediction) != record["prediction_sha256"]:
                raise ValueError("Prediction hash mismatch")
            found[key] = (record, prediction)
        except Exception as error:
            failures.append(dict(file=path.relative_to(results).as_posix(), error=str(error)))
    missing = sorted(expected - set(found))
    counts = {str(s): sum(k[0] == s for k in found) for s in SUBJECTS}
    accounting = dict(expected_fits=1350, valid_completed_fits=len(found), missing_fits=len(missing),
                      expected_participants=54, participants_with_all_25_fits=sum(n == 25 for n in counts.values()),
                      fits_by_participant=counts,
                      missing=[dict(subject=s, condition=c, seed=seed) for s, c, seed in missing],
                      invalid_records=failures, analysis_permitted=not missing and not failures,
                      test_statistics_computed=False,
                      missingness_rule="No participant is silently excluded; all 1350 fixed jobs are required")
    return accounting, found, common_identity


def verify_prediction_arrays(found):
    """All jobs must exist before arrays or test outcomes are opened here."""
    failures = []
    day_labels = {}
    for key, (record, path) in sorted(found.items()):
        subject, condition, seed = key
        try:
            with np.load(path, allow_pickle=False) as z:
                width = 2 if condition == "S0" else 16
                required = {"P_train", "P_test", "y_train", "y_test", "normalization_mean", "normalization_sd"}
                if condition != "S0":
                    required.add("Y_used_last")
                if set(z.files) != required:
                    raise ValueError("Unexpected prediction-array schema")
                for name in ("P_train", "P_test"):
                    values = z[name]
                    if values.shape != (100, width) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                        raise ValueError(f"Invalid {name}")
                for part in ("train", "test"):
                    labels = z[f"y_{part}"]
                    if labels.dtype != np.int64 or labels.shape != (100,) or np.bincount(labels, minlength=2).tolist() != [50, 50]:
                        raise ValueError(f"Invalid y_{part}")
                    label_key = (subject, part)
                    if label_key in day_labels and not np.array_equal(day_labels[label_key], labels):
                        raise ValueError(f"Different {part} labels across fits")
                    day_labels[label_key] = labels.copy()
                mean, sd = z["normalization_mean"], z["normalization_sd"]
                if mean.shape != (1, 62, 1) or sd.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(sd).all() or not (sd > 0).all():
                    raise ValueError("Invalid training-fitted normalization arrays")
                if condition != "S0":
                    target = z["Y_used_last"]
                    if target.shape != (100, 16) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
                        raise ValueError("Invalid last-used target array")
                for metric in ("acc_test", "nll_test", "ece_test"):
                    value = record[metric]
                    if not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError(f"Invalid {metric}")
                if not (0 <= record["acc_test"] <= 1 and record["nll_test"] >= 0 and 0 <= record["ece_test"] <= 1):
                    raise ValueError("Metric outside its valid range")
        except Exception as error:
            failures.append(dict(subject=subject, condition=condition, seed=seed, error=str(error)))
    return failures


def compute_final(found):
    participant_rows = []
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            records = [found[(subject, condition, seed)][0] for seed in SEEDS]
            participant_rows.append(dict(subject=subject, condition=condition, seeds=5,
                                         accuracy=float(np.mean([r["acc_test"] for r in records])),
                                         nll=float(np.mean([r["nll_test"] for r in records])),
                                         ece=float(np.mean([r["ece_test"] for r in records]))))
    lookup = {(r["subject"], r["condition"]): r for r in participant_rows}
    condition_rows = []
    for condition in CONDITIONS:
        rows = [lookup[(s, condition)] for s in SUBJECTS]
        accuracy = np.array([r["accuracy"] for r in rows])
        condition_rows.append(dict(condition=condition, n_participants=54,
                                   accuracy_mean_percent=100 * float(accuracy.mean()),
                                   accuracy_sd_between_participants_percentage_points=100 * float(accuracy.std(ddof=1)),
                                   nll_mean=float(np.mean([r["nll"] for r in rows])),
                                   ece_mean=float(np.mean([r["ece"] for r in rows])),
                                   nll_ece_scope="secondary descriptive, not additional confirmatory tests"))
    contrasts = []
    for first, second in CONTRASTS:
        one = [lookup[(s, first)]["accuracy"] for s in SUBJECTS]
        two = [lookup[(s, second)]["accuracy"] for s in SUBJECTS]
        contrasts.append(dict(contrast=f"{first}-{second}", primary=first == "W2" and second == "W0",
                              **paired_summary(one, two)))
    adjusted = holm_adjust([r["p_two_sided"] for r in contrasts])
    for row, adjusted_p in zip(contrasts, adjusted):
        row.update(p_holm_five=float(adjusted_p), holm_family_size=5,
                   significant_after_holm_0_05=bool(adjusted_p < 0.05))
    return participant_rows, condition_rows, contrasts


def read_csp_descriptive(path):
    """Optional per-participant CSP–LDA accuracy; never enters the five-test family."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = {}
    for row in rows:
        subject = int(row["subject"])
        value = float(row.get("acc_test", row.get("accuracy", "nan")))
        if subject not in SUBJECTS or subject in records or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("CSP CSV must contain one finite fractional accuracy for every participant")
        records[subject] = value
    if set(records) != set(SUBJECTS):
        raise ValueError("CSP CSV must include all 54 participants")
    values = np.array([records[s] for s in SUBJECTS])
    return dict(condition="CSP-LDA", n_participants=54,
                accuracy_mean_percent=100 * float(values.mean()),
                accuracy_sd_between_participants_percentage_points=100 * float(values.std(ddof=1)),
                scope="descriptive reference only; no added primary test", source_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csp-csv", type=Path)
    args = parser.parse_args()
    accounting, found, identity = audit_complete(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "completion_accounting.json", accounting)
    if not accounting["analysis_permitted"]:
        print(json.dumps(dict(status="INCOMPLETE_NO_INTERIM_ANALYSIS", expected_fits=1350,
                              valid_completed_fits=accounting["valid_completed_fits"],
                              missing_fits=accounting["missing_fits"],
                              invalid_records=len(accounting["invalid_records"]),
                              test_statistics_computed=False)))
        return
    array_failures = verify_prediction_arrays(found)
    if array_failures:
        accounting.update(analysis_permitted=False, invalid_prediction_arrays=array_failures)
        atomic_json(args.out / "completion_accounting.json", accounting)
        raise ValueError("Prediction-array validation failed; no statistical analysis produced")
    participants, conditions, contrasts = compute_final(found)
    reference = read_csp_descriptive(args.csp_csv) if args.csp_csv else None
    write_csv(args.out / "participant_seed_averages.csv", participants)
    write_csv(args.out / "condition_summary.csv", conditions)
    write_csv(args.out / "five_paired_contrasts_holm.csv", contrasts)
    final = dict(status="COMPLETE_FINAL_ANALYSIS", dataset="Lee2019_MI", participants=54, seeds_per_condition=5,
                 fits=1350, independent_unit="participant, after averaging five seeds",
                 protocol_identity=identity, comparisons=contrasts, descriptive_conditions=conditions,
                 csp_lda_descriptive_reference=reference,
                 confidence_interval_scope="pointwise unadjusted 95%; p-values use Holm across exactly five contrasts",
                 old_dataset_results_pooled=False,
                 analysis_code_sha256=sha(Path(__file__)),
                 input_record_sha256={p.relative_to(args.results.resolve()).as_posix(): sha(p)
                                      for p in sorted(args.results.resolve().rglob("records/*.json"))})
    atomic_json(args.out / "final_analysis.json", final)
    accounting.update(test_statistics_computed=True, prediction_arrays_verified=1350)
    atomic_json(args.out / "completion_accounting.json", accounting)
    print(json.dumps(dict(status=final["status"], participants=54, fits=1350, contrasts=5,
                          result=str((args.out / "final_analysis.json").resolve()))))


if __name__ == "__main__":
    main()
