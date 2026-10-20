"""Fixed six-feature, two-ended CSP + default SVD LDA reference for Lee2019.

This is a NEW contextual reference. Unlike the archived three-feature,
one-ended CSP reference, it explicitly takes three filters from each end.
The old study files/results are not modified. All spatial filters and LDA
parameters are fitted on day 1; day 2 is transformed and scored once.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from prepare_lee2019 import atomic_json, file_hashes


REFERENCE_ID = "CSP6_TWO_ENDED_LDA_SVD_LEE"


def validate_completed_result(data_dir: Path, out_dir: Path, subject: int, protocol: dict, script_sha256: str) -> bool:
    """Reuse only a complete result with the same frozen identity and bytes."""
    row_path = out_dir / f"subject_{subject:02d}_classical.json"
    prediction_path = out_dir / f"subject_{subject:02d}_classical_predictions.npz"
    if not row_path.exists():
        return False
    row = json.loads(row_path.read_text(encoding="utf-8"))
    if row.get("reference_id") != REFERENCE_ID or row.get("dataset") != "Lee2019_MI" or row.get("subject") != subject:
        raise RuntimeError(f"Completed reference has an incompatible identity: subject {subject}")
    if row.get("protocol") != protocol or row.get("script_sha256") != script_sha256:
        raise RuntimeError(f"Completed reference protocol or code changed: subject {subject}")
    if (row.get("n_train"), row.get("n_test"), row.get("n_features")) != (100, 100, 6):
        raise RuntimeError(f"Completed reference dimensions changed: subject {subject}")
    if row.get("lda", {}).get("classes") != [0, 1]:
        raise RuntimeError(f"Completed reference class order is invalid: subject {subject}")
    for split in ("train", "test"):
        for metric in ("acc", "nll", "ece"):
            value = row.get(f"{metric}_{split}")
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise RuntimeError(f"Completed reference has an invalid metric field: subject {subject}")
            if metric in ("acc", "ece") and value > 1:
                raise RuntimeError(f"Completed reference has an out-of-range metric field: subject {subject}")
    provenance = row["input_provenance"]
    for name, suffix in (("metadata", "meta.json"), ("train", "train.npz"), ("test", "test.npz")):
        input_path = data_dir / f"subject_{subject:02d}_{suffix}"
        if provenance[name]["filename"] != input_path.name or file_hashes(input_path)["sha256"] != provenance[name]["sha256"]:
            raise RuntimeError(f"Completed reference input bytes changed: subject {subject}, {name}")
    if not prediction_path.is_file() or row["predictions"]["filename"] != prediction_path.name:
        raise RuntimeError(f"Completed reference prediction file is missing: subject {subject}")
    if file_hashes(prediction_path)["sha256"] != row["predictions"]["sha256"]:
        raise RuntimeError(f"Completed reference prediction checksum changed: subject {subject}")
    return True


def preserve_orphan_prediction(out_dir: Path, subject: int) -> Path | None:
    """Keep an interrupted prediction export before recomputing the same job."""
    row_path = out_dir / f"subject_{subject:02d}_classical.json"
    prediction_path = out_dir / f"subject_{subject:02d}_classical_predictions.npz"
    if row_path.exists() or not prediction_path.exists():
        return None
    recovery = out_dir / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = recovery / f"{prediction_path.name}.orphan-{stamp}"
    target.resolve().relative_to(out_dir.resolve())
    checksum = file_hashes(prediction_path)
    os.replace(prediction_path, target)
    atomic_json(target.with_suffix(target.suffix + ".json"), {
        "reason": "Prediction export existed without its completed participant JSON; retained before same-protocol recomputation.",
        "subject": subject, "original_filename": prediction_path.name, "preserved_filename": target.name,
        "hashes": checksum, "preserved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    return target


def fit_csp(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original covariance/eigen arithmetic, with six explicit endpoint filters."""
    if X_train.ndim != 3 or X_train.shape[1] < 6:
        raise ValueError("CSP needs at least six EEG channels")
    if set(np.unique(y_train)) != {0, 1} or len(y_train) != len(X_train):
        raise ValueError("CSP requires semantic labels 0=left and 1=right")
    cov1 = np.mean([np.cov(trial) for trial in X_train[y_train == 0]], axis=0)
    cov2 = np.mean([np.cov(trial) for trial in X_train[y_train == 1]], axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov1 + cov2)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = np.maximum(eigvals, 1e-10)
    P = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    _, B = np.linalg.eigh(P @ cov1 @ P.T)
    # eigh returns ascending values; descending order matches the legacy code.
    B = B[:, ::-1]
    n_channels = X_train.shape[1]
    selected = np.r_[0:3, n_channels - 3:n_channels]
    if len(selected) != 6 or len(np.unique(selected)) != 6:
        raise ValueError("Conventional CSP reference must have six unique filters")
    W = B[:, selected].T @ P
    if W.shape != (6, n_channels) or not np.isfinite(W).all():
        raise ValueError("Invalid trained spatial filters")
    return W, selected


def csp_transform(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Match the archived log population-variance arithmetic without adding a
    # fitted scaling step or tuning epsilon against held-out performance.
    features = np.array([np.log(np.var(W @ trial, axis=1)) for trial in X])
    if features.shape != (len(X), 6) or not np.isfinite(features).all():
        raise ValueError("CSP produced non-finite or incorrectly shaped features")
    return features


def class_metrics(prob, labels):
    """Same 15-bin ECE and clipped NLL definitions as released EEGNet core."""
    prob = prob.astype(np.float64)
    if prob.shape != (len(labels), 2) or not np.isfinite(prob).all():
        raise ValueError("Expected finite probabilities for left/right columns")
    if np.any(prob < 0) or not np.allclose(prob.sum(1), 1.0, atol=1e-10):
        raise ValueError("LDA probabilities must be nonnegative and sum to one")
    pred, conf = prob.argmax(1), prob.max(1)
    ok = (pred == labels).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for i in range(15):
        mask = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            ece += mask.mean() * abs(ok[mask].mean() - conf[mask].mean())
    return {"acc": float(ok.mean()), "nll": float(-np.log(np.clip(prob[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()), "ece": float(ece)}


def load_subject(data_dir: Path, subject: int) -> tuple:
    stem = f"subject_{subject:02d}"
    meta_path = data_dir / f"{stem}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset") != "Lee2019_MI" or meta.get("subject") != subject:
        raise ValueError("Subject/dataset metadata mismatch")
    if meta.get("data_units") != "uV" or meta.get("sampling_hz") != 128.0 or meta.get("label_mapping") != {"left_hand": 0, "right_hand": 1}:
        raise ValueError("Reference requires the frozen microvolt, 128 Hz semantic-label contract")
    if len(meta.get("channel_names", [])) != 62 or not meta.get("held_out_day2_prepared"):
        raise ValueError("Both separate days and all 62 EEG channels are required")
    arrays, provenance = {}, {"metadata": {"filename": meta_path.name, **file_hashes(meta_path)}}
    for split, day in (("train", 1), ("test", 2)):
        npz_path = data_dir / f"{stem}_{split}.npz"
        hashes = file_hashes(npz_path)
        day_meta = meta["days"][split]
        if day_meta["source_session"] != day or day_meta["source_run"] != "EEG_MI_train":
            raise ValueError("Reference split must be offline day 1 -> offline day 2")
        if not day_meta["source"]["published_md5_verified"] or hashes["sha256"] != day_meta["prepared_npz"]["sha256"]:
            raise ValueError("Input checksum disagrees with verified preparation metadata")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {f"X_{split}", f"y_{split}"}:
                raise ValueError("Unexpected NPZ keys")
            X, y = archive[f"X_{split}"], archive[f"y_{split}"]
        if X.dtype != np.float32 or X.shape != (100, 62, 512) or not np.isfinite(X).all():
            raise ValueError("Reference requires complete 100 x 62 x 512 prepared EEG")
        classes, counts = np.unique(y, return_counts=True)
        if y.dtype != np.int64 or y.shape != (100,) or classes.tolist() != [0, 1] or counts.tolist() != [50, 50]:
            raise ValueError("Expected 50 left/50 right offline labels")
        arrays[split] = (X, y)
        provenance[split] = {"filename": npz_path.name, **hashes, "source_sha256": day_meta["source"]["sha256"]}
    return arrays["train"], arrays["test"], provenance


def run_subject(data_dir: Path, out_dir: Path, subject: int, protocol: dict, script_sha256: str) -> None:
    started = time.perf_counter()
    row_path = out_dir / f"subject_{subject:02d}_classical.json"
    prediction_path = out_dir / f"subject_{subject:02d}_classical_predictions.npz"
    if validate_completed_result(data_dir, out_dir, subject, protocol, script_sha256):
        print(json.dumps({"status": "reference_subject_reused_verified", "subject": subject, "record": str(row_path)}), flush=True)
        return
    preserve_orphan_prediction(out_dir, subject)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    (Xtr, ytr), (Xte, yte), input_provenance = load_subject(data_dir, subject)
    W, indices = fit_csp(Xtr, ytr)
    Ftr, Fte = csp_transform(Xtr, W), csp_transform(Xte, W)
    lda = LinearDiscriminantAnalysis()  # solver='svd', shrinkage=None, priors=None
    lda.fit(Ftr, ytr)
    if lda.classes_.tolist() != [0, 1]:
        raise ValueError("LDA probability columns must be left=0 then right=1")
    ptr, pte = lda.predict_proba(Ftr), lda.predict_proba(Fte)
    train, test = class_metrics(ptr, ytr), class_metrics(pte, yte)
    prediction_tmp = prediction_path.with_suffix(".tmp.npz")
    np.savez_compressed(prediction_tmp, y_train=ytr, y_test=yte, P_train=ptr, P_test=pte,
                        W_train_only=W, selected_filter_indices=indices,
                        F_train=Ftr, F_test=Fte, lda_classes=lda.classes_,
                        lda_coef=lda.coef_, lda_intercept=lda.intercept_)
    os.replace(prediction_tmp, prediction_path)
    row = {
        "reference_id": REFERENCE_ID, "dataset": "Lee2019_MI", "subject": subject,
        "n_train": len(ytr), "n_test": len(yte), "n_features": 6,
        "acc_train": train["acc"], "nll_train": train["nll"], "ece_train": train["ece"],
        "acc_test": test["acc"], "nll_test": test["nll"], "ece_test": test["ece"],
        "csp": {"selection": "three largest and three smallest whitened class-0 covariance eigenvalues",
                "selected_zero_based_indices": indices.tolist(), "covariance": "mean of np.cov(trial) within each training class",
                "whitening_eigenvalue_floor": 1e-10, "features": "log(np.var(W @ trial, axis=1)); six features",
                "fit_data": "day 1 only", "new_reference_differs_from_legacy_three_feature_one_ended_CSP": True},
        "lda": {"class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
                "parameters": lda.get_params(), "classes": lda.classes_.tolist(), "fit_data": "day 1 only"},
        "additional_feature_scaling": None, "hyperparameter_search": None,
        "role": "secondary descriptive contextual reference; no new confirmatory contrast family",
        "protocol": protocol, "input_provenance": input_provenance,
        "script_sha256": script_sha256,
        "predictions": {"filename": prediction_path.name, **file_hashes(prediction_path)},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(row_path, row)
    # Progress only: no condition comparison or outcome-based steering.
    print(json.dumps({"status": "reference_subject_complete", "subject": subject, "record": str(row_path)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True, help="Frozen protocol digest, checked before any test input is read")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 55)))
    args = parser.parse_args(argv)
    if len(set(args.subjects)) != len(args.subjects) or any(subject not in range(1, 55) for subject in args.subjects):
        raise ValueError("Subjects must be unique IDs 1..54")
    digest = file_hashes(args.protocol)["sha256"]
    if digest != args.protocol_sha256.lower():
        raise ValueError("Protocol digest changed; do not read held-out inputs")
    protocol = {"filename": args.protocol.name, "sha256": digest}
    script_sha256 = file_hashes(Path(__file__))["sha256"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        run_subject(args.data_dir, args.out_dir, subject, protocol, script_sha256)
    all_record_paths = [args.out_dir / f"subject_{subject:02d}_classical.json" for subject in range(1, 55)]
    if all(path.is_file() for path in all_record_paths):
        for subject in range(1, 55):
            validate_completed_result(args.data_dir, args.out_dir, subject, protocol, script_sha256)
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in all_record_paths]
        if any(row["protocol"] != protocol or row["script_sha256"] != script_sha256 for row in rows):
            raise ValueError("Do not aggregate reference results from mixed protocols or scripts")
        csv_path = args.out_dir / "classical_summary.csv"
        csv_tmp = csv_path.with_suffix(".tmp.csv")
        keys = ["dataset", "subject", "n_train", "n_test", "n_features", "acc_train", "nll_train", "ece_train", "acc_test", "nll_test", "ece_test"]
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row[key] for key in keys} for row in rows)
        os.replace(csv_tmp, csv_path)
        atomic_json(args.out_dir / "classical_reference_manifest.json", {
            "reference_id": REFERENCE_ID, "protocol": protocol, "script_sha256": script_sha256,
            "all_54_subjects_complete": True,
            "summary": {"filename": csv_path.name, **file_hashes(csv_path)},
            "per_subject_records": [{"filename": path.name, **file_hashes(path)} for path in all_record_paths],
            "prediction_files": [row["predictions"] for row in rows],
        })
    print(json.dumps({"status": "reference_requested_subjects_complete", "subjects": args.subjects}), flush=True)


if __name__ == "__main__":
    main()
