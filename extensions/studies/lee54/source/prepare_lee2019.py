"""Prepare one Lee2019-MI day, without opening the other day or online run.

Matches the MOABB 1.5 signal-processing order: microvolts -> volts;
continuous 4-38 Hz IIR filtering; inclusive 0-4 s epochs; 128 Hz resampling;
return microvolts. No fitted normalization or model selection takes place.

The upstream name ``EEG_MI_train`` means the labelled offline recording on
each day. Here day 1 becomes our training set and day 2 our held-out test set.
The upstream online ``EEG_MI_test`` is never loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


DATASET = "Lee2019_MI"
SOURCE_BASE = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"
CLASS_MAP = {"left": 2, "right": 1}
LABEL_MAP = {"left_hand": 0, "right_hand": 1}
SCHEMA = "lee2019-separated-days-v1"


def file_hashes(path: Path) -> dict[str, str]:
    hashes = {"sha256": hashlib.sha256(), "md5": hashlib.md5()}
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            for hasher in hashes.values():
                hasher.update(block)
    return {name: hasher.hexdigest() for name, hasher in hashes.items()}


def checksum_entry(manifest: Path, subject: int, day: int) -> str:
    filename = f"sess{day:02d}_subj{subject:02d}_EEG_MI.mat"
    target = f"session{day}/s{subject}/{filename}"
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and fields[1].lstrip("*./").replace("\\", "/") == target:
            return fields[0].lower()
    raise ValueError(f"Official MD5 manifest has no entry for {target}")


def _scalar(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"Expected a scalar, found shape {value.shape}")
    result = value.item()
    while isinstance(result, np.ndarray):
        if result.size != 1:
            raise ValueError("Non-scalar nested MATLAB value")
        result = result.item()
    return result


def _text(value) -> str:
    result = _scalar(value)
    return result.decode("utf-8") if isinstance(result, bytes) else str(result)


def read_offline_fields(path: Path) -> dict:
    # variable_names avoids loading EEG_MI_test and any other top-level variable.
    from scipy.io import loadmat

    mat = loadmat(str(path), variable_names=["EEG_MI_train"], squeeze_me=False)
    if "EEG_MI_train" not in mat:
        raise ValueError("MAT file has no EEG_MI_train offline recording")
    run = mat["EEG_MI_train"]
    if run.shape != (1, 1) or run.dtype.names is None:
        raise ValueError("Expected the published 1 x 1 EEG_MI_train structure")
    run = run[0, 0]
    required = {"x", "fs", "chan", "t", "y_dec", "class"}
    if not required.issubset(set(run.dtype.names)):
        raise ValueError(f"Missing offline fields: {required - set(run.dtype.names)}")
    mapping = {}
    for pair in run["class"]:
        if len(pair) != 2:
            raise ValueError("Expected code/name rows in the MATLAB class mapping")
        code, name = pair
        mapping[_text(name).lower()] = int(_scalar(code))
    return {
        "signal_uV": np.asarray(run["x"]),
        "sampling_hz": float(_scalar(run["fs"])),
        "channel_names": [_text(c) for c in np.ravel(run["chan"])],
        "event_samples": np.asarray(run["t"]).reshape(-1),
        "source_codes": np.asarray(run["y_dec"]).reshape(-1),
        "source_class_mapping": mapping,
    }


def validate_fields(fields: dict, *, expected_trials: int = 100) -> dict:
    if fields["source_class_mapping"] != CLASS_MAP:
        raise ValueError(f"Unexpected class mapping {fields['source_class_mapping']!r}")
    if fields["sampling_hz"] != 1000.0:
        raise ValueError(f"Expected source sampling frequency 1000 Hz: {fields['sampling_hz']}")
    signal = fields["signal_uV"]
    channels = fields["channel_names"]
    if signal.ndim != 2 or signal.shape[1] != 62 or len(channels) != 62:
        raise ValueError(f"Expected continuous samples x 62 EEG channels: {signal.shape}")
    if len(set(channels)) != 62 or any(not ch or ch.upper().startswith("EMG") for ch in channels):
        raise ValueError("Source EEG channel names must be 62 unique names without EMG")
    if not np.issubdtype(signal.dtype, np.number) or not np.isfinite(signal).all():
        raise ValueError("EEG signal must contain finite numeric values")
    event_samples, codes = fields["event_samples"], fields["source_codes"]
    if event_samples.shape != (expected_trials,) or codes.shape != (expected_trials,):
        raise ValueError(f"Expected {expected_trials} labelled offline trials")
    if not np.isfinite(event_samples).all() or not np.array_equal(event_samples, event_samples.astype(np.int64)):
        raise ValueError("Event indices must be finite integer sample indices")
    if not np.array_equal(codes, codes.astype(np.int64)) or set(np.unique(codes)) != {1, 2}:
        raise ValueError("Offline event codes must be exactly 1=right and 2=left")
    event_samples, codes = event_samples.astype(np.int64), codes.astype(np.int64)
    if np.any(np.diff(event_samples) <= 0):
        raise ValueError("Trial events must have unique, increasing sample indices")
    if event_samples[0] < 0 or event_samples[-1] + 4000 >= signal.shape[0]:
        raise ValueError("At least one requested 0-4 s epoch lies outside the offline recording")
    counts = {str(code): int(np.count_nonzero(codes == code)) for code in (1, 2)}
    if any(count != expected_trials // 2 for count in counts.values()):
        raise ValueError(f"Expected balanced offline labels, found {counts}")
    fields["event_samples"], fields["source_codes"] = event_samples, codes
    return fields


def preprocess(fields: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import mne

    info = mne.create_info(fields["channel_names"], fields["sampling_hz"], ch_types="eeg")
    raw = mne.io.RawArray(fields["signal_uV"].T * 1e-6, info, verbose=False)
    # MOABB 1.5 get_filter_pipeline: IIR filtering of continuous data, default
    # MNE forward/backward fourth-order Butterworth. Parameters recorded below.
    raw.filter(4.0, 38.0, method="iir", picks="data", verbose=False)
    events = np.column_stack((fields["event_samples"], np.zeros(len(fields["source_codes"]), dtype=int), fields["source_codes"]))
    epochs = mne.Epochs(
        raw, events, event_id={"left_hand": 2, "right_hand": 1},
        tmin=0.0, tmax=4.0, proj=False, baseline=None, preload=True,
        picks="eeg", event_repeated="drop", on_missing="raise", verbose=False,
    )
    if len(epochs) != len(events) or not np.array_equal(epochs.events, events):
        raise ValueError("Epoching unexpectedly discarded or changed source trial events")
    source_n_times = len(epochs.times)
    epochs.resample(128.0, verbose=False)
    X = (epochs.get_data(copy=False) * 1e6).astype(np.float32)
    y = np.where(epochs.events[:, 2] == 2, 0, 1).astype(np.int64)
    if X.shape[:2] != (len(events), 62) or not np.isfinite(X).all():
        raise ValueError("Invalid processed EEG shape or values")
    details = {
        "pipeline": ["source_microvolts_to_volts", "continuous_iir_bandpass_4_38_Hz", "epoch_0_to_4_s_inclusive", "epoch_resample_128_Hz", "volts_to_microvolts", "cast_float32"],
        "filter": {"method": "iir", "iir_params": "MNE defaults: order 4, Butterworth", "phase": "zero (forward-backward)", "l_freq": 4.0, "h_freq": 38.0},
        "baseline": None, "projection": False, "referencing": "published nasion reference retained; no rereference",
        "epoch_tmin_requested_s": 0.0, "epoch_tmax_requested_s": 4.0,
        "source_epoch_samples_inclusive": source_n_times,
        "resample_hz": 128.0, "resample_method": "MNE default FFT",
        "output_epoch_samples": X.shape[2], "output_last_sample_time_s": float(epochs.times[-1]),
        "normalization": "none; any train-fitted normalization belongs to the training runner",
        "event_index_policy": "Use the published t indices directly, matching MOABB Lee2019; no subtraction of one",
        "excluded": ["EEG_MI_test online recording", "EMG channels", "resting-state variables", "other day"],
        "trial_rejection": "none; abort if MNE drops a trial",
    }
    return X, y, details


def versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "mne", "moabb"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 55), metavar="1..54")
    parser.add_argument("--day", type=int, required=True, choices=(1, 2))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checksum-manifest", type=Path, default=Path(__file__).with_name("100542.md5"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    expected_name = f"sess{args.day:02d}_subj{args.subject:02d}_EEG_MI.mat"
    if args.mat.name != expected_name:
        raise ValueError(f"Source filename must identify the requested subject/day: {expected_name}")
    split = "train" if args.day == 1 else "test"
    target = args.out_dir / f"subject_{args.subject:02d}_{split}.npz"
    meta_path = args.out_dir / f"subject_{args.subject:02d}_meta.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Will not replace existing preparation: {target}")
    hashes = file_hashes(args.mat)
    expected_md5 = checksum_entry(args.checksum_manifest, args.subject, args.day)
    if hashes["md5"] != expected_md5:
        raise ValueError(f"Downloaded source fails the official MD5: expected {expected_md5}, found {hashes['md5']}")
    if args.expected_sha256 and hashes["sha256"] != args.expected_sha256.lower():
        raise ValueError("Downloaded source SHA256 differs from the recorded value")
    checksum_seconds = time.perf_counter() - started
    stage = time.perf_counter()
    fields = validate_fields(read_offline_fields(args.mat))
    load_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    X, y, processing = preprocess(fields)
    process_seconds = time.perf_counter() - stage
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "schema": SCHEMA, "dataset": DATASET, "subject": args.subject,
        "data_units": "uV", "sampling_hz": 128.0, "sfreq": 128.0,
        "channel_names": fields["channel_names"], "label_mapping": LABEL_MAP,
        "normalization": "none", "days": {},
    }
    if meta["schema"] != SCHEMA or meta["subject"] != args.subject or meta["channel_names"] != fields["channel_names"]:
        raise ValueError("Existing subject metadata has an incompatible schema, subject or channel order")
    opposite = "test" if split == "train" else "train"
    if opposite in meta["days"] and meta["days"][opposite]["shape"][1:] != list(X.shape[1:]):
        raise ValueError("Train and held-out day have incompatible channel/sample dimensions")
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"X_{split}": X, f"y_{split}": y})
    os.replace(tmp, target)
    meta["days"][split] = {
        "source_session": args.day, "source_run": "EEG_MI_train", "split": split,
        "source_class_mapping": fields["source_class_mapping"],
        "source_event_codes_to_labels": {"2": 0, "1": 1},
        "class_counts": {"left_hand": int((y == 0).sum()), "right_hand": int((y == 1).sum())},
        "shape": list(X.shape), "X_dtype": str(X.dtype), "y_dtype": str(y.dtype),
        "event_sample_indices_source": fields["event_samples"].tolist(),
        "source": {"filename": args.mat.name, "relative_path": f"session{args.day}/s{args.subject}/{args.mat.name}",
                   "url": f"{SOURCE_BASE}session{args.day}/s{args.subject}/{args.mat.name}",
                   "size_bytes": args.mat.stat().st_size, **hashes, "published_md5": expected_md5,
                   "published_md5_verified": True, "dataset_doi": "10.5524/100542", "license": "CC0-1.0"},
        "processing": processing, "versions": versions(),
        "preparation_script_sha256": file_hashes(Path(__file__))["sha256"],
        "prepared_npz": {"filename": target.name, "size_bytes": target.stat().st_size, **file_hashes(target)},
        "timing_seconds": {"source_checksum": checksum_seconds, "read_validate_offline": load_seconds,
                           "filter_epoch_resample": process_seconds, "total": time.perf_counter() - started},
        "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    meta["held_out_day2_prepared"] = "test" in meta["days"]
    atomic_json(meta_path, meta)
    print(json.dumps({"status": "prepared", "npz": str(target), "metadata": str(meta_path), "split": split,
                      "shape": list(X.shape), "class_counts": meta["days"][split]["class_counts"],
                      "timing_seconds": meta["days"][split]["timing_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
