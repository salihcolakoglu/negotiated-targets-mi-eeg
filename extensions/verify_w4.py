#!/usr/bin/env python3
"""Verify the completed Lee W4 archive and its historical W1/W2 pairing.

This separate verifier leaves the original three-study verifier unchanged.
It reads retained test outputs only: no EEG, Torch, checkpoints or fitting.
The old Lee bit clipping (1e-7), class floor (1e-12), and float64 native
Euclidean decoder are intentional parts of the recorded experiment.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import zipfile
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("W1", "W2", "W4")
METRICS = ("accuracy", "nll_clipped", "ece15", "bce", "brier_sum",
           "mean_confidence", "posterior_accuracy", "train_accuracy",
           "target_displacement", "train_seconds")
COUNTS = dict(new_fits=270, new_tests=270, historical_comparators=540,
              model_artifacts=864, test_artifacts=540, participant_means=162)
PROTOCOL_SHA = "1702e81efcc5dcadb04546acdcac6be90645aaac4b56fe238e57f939d73f49d4"


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, label, atol=2e-10):
    if a is None or b is None:
        require(a is None and b is None, label + ": null eligibility")
    else:
        require(math.isfinite(float(a)) and math.isfinite(float(b)) and
                abs(float(a) - float(b)) <= atol, label + ": mismatch")


def local(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name,
            "Canonical relative public path required")
    q = (root / p).resolve()
    require(q.is_relative_to(root.resolve()), "Public path escaped repository")
    return q


def load_json(raw):
    return json.loads(raw.decode("utf-8"))


def checked_zip(z, members):
    names = z.namelist()
    require(len(names) == len(set(names)) and set(names) == set(members),
            "Missing, duplicate or orphan archive member")
    for name in names:
        require(not name.startswith(("/", "\\")) and ".." not in Path(name).parts
                and "\\" not in name, "Unsafe member path")
        require(digest(z.read(name)) == members[name], "Member SHA mismatch: " + name)


def metric(P, y):
    require(P.dtype == np.float32 and P.shape == (100, 16), "Float32 100x16 sigmoid output required")
    require(y.dtype == np.int64 and y.shape == (100,) and np.isin(y, [0, 1]).all()
            and np.bincount(y, minlength=2).tolist() == [50, 50], "Balanced int64 100-label order required")
    require(np.isfinite(P).all() and ((0 <= P) & (P <= 1)).all(), "Invalid network outputs")
    C = np.array([[1.] * 16, [1., 0.] * 8], dtype=np.float64)
    p = P.astype(np.float64)
    # Preserve historical finite-precision operation order, including ties.
    native = ((p[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logs = np.log(clipped) @ C.T + np.log(1 - clipped) @ (1 - C.T)
    logs -= logs.max(1, keepdims=True)
    posterior = np.exp(logs)
    posterior /= posterior.sum(1, keepdims=True)
    decision = posterior.argmax(1)
    confidence = posterior.max(1)
    edges = np.linspace(0., 1., 16)
    calibration = 0.
    for k in range(15):
        lower = confidence >= edges[k] if k == 0 else confidence > edges[k]
        group = lower & (confidence <= edges[k + 1])
        if group.any():
            calibration += group.mean() * abs((decision[group] == y[group]).mean() - confidence[group].mean())
    target = C[y]
    return dict(accuracy=float((native == y).mean()),
                nll_clipped=float(-np.log(np.clip(posterior[np.arange(100), y], 1e-12, 1)).mean()),
                ece15=float(calibration),
                bce=float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()),
                brier_sum=float(((posterior - np.eye(2)[y]) ** 2).sum(1).mean()),
                mean_confidence=float(confidence.mean()),
                posterior_accuracy=float((decision == y).mean()))


def paired(values):
    d = np.asarray(values, dtype=np.float64)
    require(d.shape == (54,) and np.isfinite(d).all(), "Exactly 54 paired person means required")
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        ci = [mu, mu]
        statistic = 0. if mu == 0 else None
        probability = 1. if mu == 0 else 0.
        effect = 0. if mu == 0 else None
    else:
        statistic, probability = map(float, stats.ttest_1samp(d, 0.))
        ci = [float(x) for x in stats.t.interval(.95, df=53, loc=mu, scale=stats.sem(d))]
        effect = mu / sd
    return dict(mean_difference_pp=mu, pointwise_95ci_pp=ci, sd_difference_pp=sd,
                t=statistic, df=53, p_unadjusted=probability, d_z=effect,
                participants_improved=int((d > 0).sum()), participants_worse=int((d < 0).sum()),
                participants_tied=int((d == 0).sum()), n=54)


def holm_two(p, q):
    require(0 <= p <= 1 and 0 <= q <= 1, "Invalid p values")
    if p <= q:
        return [min(1., 2 * p), min(1., max(2 * p, q))]
    return [min(1., max(p, 2 * q)), min(1., 2 * q)]


def summarize(rows):
    require(len(rows) == 810, "810 paired model records, not 810 people")
    people = {}
    for s in range(1, 55):
        for c in CONDITIONS:
            group = [r for r in rows if r["subject"] == s and r["condition"] == c]
            require(len(group) == 5 and {r["seed"] for r in group} == set(range(1, 6)), "Five paired starts per person")
            people[s, c] = {k: float(np.mean([r[k] for r in group])) for k in METRICS}
    summaries = [dict(condition=c, n_participants=54, starts_per_participant=5,
                      **{k: float(np.mean([people[s, c][k] for s in range(1, 55)])) for k in METRICS})
                 for c in CONDITIONS]
    contrasts, differences = [], {}
    for label, left, right in (("W2-W4", "W2", "W4"), ("W4-W1", "W4", "W1")):
        d = [round(100 * (people[s, left]["accuracy"] - people[s, right]["accuracy"]), 10) for s in range(1, 55)]
        contrasts.append(dict(contrast=label, primary=label == "W2-W4", **paired(d)))
        for s in range(1, 55):
            differences[s, label] = dict(accuracy_difference_pp=d[s - 1],
                **{k + "_difference": people[s, left][k] - people[s, right][k] for k in METRICS if k != "accuracy"})
    for row, adjusted in zip(contrasts, holm_two(*[r["p_unadjusted"] for r in contrasts])):
        row["p_holm_two"] = adjusted
    return people, summaries, contrasts, differences


def compare_rows(observed, expected, fields, label):
    require(set(observed) == set(expected), label + ": missing or orphan identity")
    for key in expected:
        for field in fields:
            value, retained = expected[key][field], observed[key][field]
            if isinstance(value, list):
                require(len(value) == len(retained), label + ": interval length")
                for a, b in zip(value, retained):
                    close(a, b, label + "/" + field)
            else:
                close(value, retained, label + "/" + field)


def csv_rows(z, name):
    return list(csv.DictReader(io.StringIO(z.read("tables/" + name).decode("utf-8-sig"))))


def verify(repository=ROOT):
    root = Path(repository).resolve()
    manifest = load_json((root / "extensions/W4_RESULTS_MANIFEST.json").read_bytes())
    require(manifest["status"] == "COMPLETE_INDEPENDENTLY_AUDITED_W4_RESULTS_INCLUDED"
            and manifest["source_protocol_sha256"] == PROTOCOL_SHA and manifest["counts"] == COUNTS,
            "Completed, fixed-protocol W4 manifest required")
    archive = local(root, manifest["archive"])
    base_archive = local(root, manifest["base_archive"])
    base_manifest_path = local(root, manifest["base_manifest"])
    require(file_sha(archive) == manifest["archive_sha256"] and
            file_sha(base_archive) == manifest["base_archive_sha256"] and
            file_sha(base_manifest_path) == manifest["base_manifest_sha256"], "Archive/metadata byte gate")
    base_manifest = load_json(base_manifest_path.read_bytes())
    require(base_manifest["archive_sha256"] == manifest["base_archive_sha256"], "Historical base archive binding")
    for rel, h in manifest["public_kernel_sha256"].items():
        require(file_sha(local(root, rel)) == h, "Preserved W4 source changed: " + rel)
    with zipfile.ZipFile(archive) as new, zipfile.ZipFile(base_archive) as base:
        # All byte gates precede the first NumPy decode.
        checked_zip(new, manifest["members"])
        checked_zip(base, base_manifest["members"])
        index = load_json(new.read("W4_INDEX.json"))
        require(index["schema"] == "public-w4-completed-v1" and index["counts"] == COUNTS,
                "W4 index schema/counts")
        runs = index["runs"]
        expected_ids = {f"subject_{s:02d}_train__{c}__seed{k}": (s, c, k)
                        for s in range(1, 55) for c in CONDITIONS for k in range(1, 6)}
        require(len(runs) == len(expected_ids) == 810 and {r["id"] for r in runs} == set(expected_ids), "Exact W1/W2/W4 identities")
        histories = {r["id"]: r for r in load_json(base.read("RUN_INDEX.json"))["runs"]
                     if r["study"] == "lee54" and r["condition"] in ("W1", "W2")}
        require(len(histories) == 540, "All historical W1/W2 records required")
        by_id = {r["id"]: r for r in runs}
        referenced = {"W4_INDEX.json", "SOURCE_AUDIT.json"} | {"tables/" + n for n in
            ("ANCHOR_GATE_RESULTS-V1.0.json", "run_metrics.csv", "participant_means.csv", "condition_summary.csv",
             "participant_differences.csv", "two_contrasts_holm.json")}
        for fid, (s, c, k) in expected_ids.items():
            r = by_id[fid]
            require((r["subject"], r["condition"], r["seed"], r["n"]) == (s, c, k, 100), "Run identity/budget")
            require(r["fit_metadata"]["n_train"] == 100 and r["fit_metadata"]["epochs_completed"] == 300,
                    "Matched 100-trial/300-epoch budgets")
            if c == "W4":
                require(r["prediction_archive"] == "w4" and r["prediction_file"] == "predictions/" + fid + ".npz", "Owned W4 output path")
                require(r["fit_metadata"]["actual_update_calls"] == 2100, "Recorded terminal W4 update budget")
                referenced.add(r["prediction_file"])
            else:
                original = histories[fid]
                require(r["prediction_archive"] == "base", "Historical prediction archive role")
                for field in ("subject", "condition", "seed", "prediction_file", "prediction_sha256", "source_prediction_sha256", "source_record_sha256"):
                    require(r[field] == original[field], "Historical pairing changed: " + field)
            require(set(r["prepared_input_sha256"]) == {"train", "test", "meta"}, "Explicit same-input byte witnesses")
            for val in r["prepared_input_sha256"].values():
                require(isinstance(val, str) and len(val) == 64 and all(c in "0123456789abcdef" for c in val), "Invalid SHA witness")
        require(set(new.namelist()) == referenced and len(referenced) == 278, "Exactly 270 new P/y files + eight metadata/tables")
        audit = load_json(new.read("SOURCE_AUDIT.json"))
        require(audit["status"] == "PASS_INDEPENDENT_FINAL_W4_QA" and audit["protocol_sha256"] == PROTOCOL_SHA
                and audit["results_sha256"] == manifest["source_results_sha256"]
                and audit["completion_sha256"] == manifest["source_completion_sha256"], "Source independent complete audit")
        labels, inputs, rows = {}, {}, []
        for fid in sorted(expected_ids):
            r = by_id[fid]
            z = new if r["prediction_archive"] == "w4" else base
            raw = z.read(r["prediction_file"])
            require(digest(raw) == r["prediction_sha256"], "Retained run SHA: " + fid)
            with np.load(io.BytesIO(raw), allow_pickle=False) as a:
                require(set(a.files) == {"P_test", "y_test"}, "Test outputs only; no EEG/train/checkpoint members")
                y, P = a["y_test"], a["P_test"]
                observed = metric(P, y)
            prior = labels.setdefault(r["subject"], y.copy())
            require(np.array_equal(y, prior), "Original test row order must match all paired starts/conditions")
            prior_input = inputs.setdefault(r["subject"], r["prepared_input_sha256"])
            require(prior_input == r["prepared_input_sha256"], "Prepared inputs must be identical across paired conditions")
            for field, value in observed.items():
                close(value, r["metrics"][field], "Retained run " + fid + "/" + field)
            rows.append(dict(subject=r["subject"], seed=r["seed"], condition=r["condition"], **observed,
                             **{k: r["fit_metadata"][k] for k in METRICS[-3:]}))
        people, summaries, contrasts, differences = summarize(rows)
        rcsv = csv_rows(new, "run_metrics.csv")
        ri = {(int(r["subject"]), r["condition"], int(r["seed"])): r for r in rcsv}
        er = {(r["subject"], r["condition"], r["seed"]): r for r in rows}
        require(len(ri) == len(rcsv) == 810, "No duplicate run CSV row")
        compare_rows(ri, er, METRICS, "Run CSV")
        pcsv = csv_rows(new, "participant_means.csv")
        pi = {(int(r["subject"]), r["condition"]): r for r in pcsv}
        require(len(pi) == len(pcsv) == 162, "No duplicate participant row")
        compare_rows(pi, people, METRICS, "54-person seed means")
        scsv = csv_rows(new, "condition_summary.csv")
        si = {r["condition"]: r for r in scsv}
        require(len(si) == len(scsv) == 3, "Three condition CSV rows")
        compare_rows(si, {r["condition"]: r for r in summaries}, METRICS, "Condition CSV")
        result = load_json(new.read("tables/ANCHOR_GATE_RESULTS-V1.0.json"))
        require(result["status"] == "COMPLETE_ANCHOR_GATE_ABLATION_ANALYSIS_VERIFIED" and result["counts"] == COUNTS
                and result["protocol_sha256"] == PROTOCOL_SHA, "Source results identity")
        require(digest(new.read("tables/ANCHOR_GATE_RESULTS-V1.0.json")) == manifest["source_results_sha256"], "Unmodified source results JSON")
        compare_rows({r["condition"]: r for r in result["summaries"]}, {r["condition"]: r for r in summaries},
                     METRICS + ("n_participants", "starts_per_participant"), "Condition JSON")
        stat_fields = tuple(k for k in contrasts[0] if k not in ("contrast", "primary"))
        for recorded in (result["contrasts"], load_json(new.read("tables/two_contrasts_holm.json"))):
            require(len(recorded) == 2 and [(r["contrast"], r["primary"]) for r in recorded] == [("W2-W4", True), ("W4-W1", False)],
                    "Exactly two planned new accuracy tests")
            compare_rows({r["contrast"]: r for r in recorded}, {r["contrast"]: r for r in contrasts}, stat_fields, "Paired t/Holm")
        dcsv = csv_rows(new, "participant_differences.csv")
        di = {(int(r["subject"]), r["contrast"]): r for r in dcsv}
        require(len(di) == len(dcsv) == 108, "108 unique person contrasts")
        compare_rows(di, differences, tuple(next(iter(differences.values()))), "Person differences")
    return dict(status="PASS_PORTABLE_W4_RETAINED_VERIFICATION", new_prediction_files=270,
                historical_prediction_files_reused=540, participants=54, starts=5, participant_means=162,
                accuracy_tests=2, Holm_family_size=2, original_five_test_family_modified=False,
                source_protocol_sha256=PROTOCOL_SHA, archive_sha256=manifest["archive_sha256"],
                scope="Independent retained native/probability metrics and paired t/Holm replay; saved training metrics/timing are provenance only, not retraining proof or a controlled speed ranking.",
                limitations="Exploratory after Lee outcomes had been seen; pointwise95% intervals are not multiplicity-adjusted intervals.",
                raw_EEG_reads=0, checkpoint_reads=0, training_or_inference=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.repository)
    if args.report:
        require(not args.report.exists(), "Choose a new report path")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
