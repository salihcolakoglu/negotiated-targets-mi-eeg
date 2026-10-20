"""Lee-specific input contract; no fitting operation ever needs Day 2."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

R_RATE, PHASES = 0.03, 10
CONDITIONS = ("S0", "W0", "W1", "W2", "W3")
RULES = {"S0": "hard", "W0": "hard", "W1": "neg_paper",
         "W2": "neg_anchor_gate", "W3": "smooth_class"}
PARAMS = dict(lr=0.001, epochs_per_phase=30, phases=10, batch=16, dropout=0.5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_walsh(size_of_walsh=16):
    # Exact released implementation, including row order.
    n = int(math.log(size_of_walsh, 2) + 1)
    j = 1
    W_old = 1
    for i in range(n):
        j_new = 2 ** i
        W_new = np.zeros([j_new, j_new], dtype=np.float32)
        W_new[:j, :j] = W_old
        W_new[j:, :j] = W_old
        W_new[:j, j:] = W_old
        W_new[j:, j:] = abs(1 - W_old)
        W_old = W_new
        j = j_new
    return W_new


def code_rows(n_cls=2):
    if n_cls != 2:
        raise ValueError("Lee validation is fixed to left/right imagery")
    return create_walsh(16)[:n_cls].astype(np.float32)


def load_part(path, part):
    """Read a single offline day in microvolts, without opening another day."""
    if part not in ("train", "test"):
        raise ValueError(part)
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"X_{part}", f"y_{part}"}
        if set(archive.files) != expected:
            raise ValueError(f"Expected only {sorted(expected)} in {path}")
        X = archive[f"X_{part}"]
        y = archive[f"y_{part}"]
    if X.dtype != np.float32 or y.dtype != np.int64:
        raise ValueError(f"Unexpected array dtypes in {path}")
    if X.ndim != 3 or X.shape[1] != 62 or X.shape[2] < 32:
        raise ValueError(f"Expected trials x 62 EEG channels x time; got {X.shape}")
    if X.shape[0] != len(y) or y.shape != (len(y),) or not np.isfinite(X).all():
        raise ValueError(f"Invalid trial/label arrays in {path}")
    labels, counts = np.unique(y, return_counts=True)
    if labels.tolist() != [0, 1] or counts.tolist() != [50, 50]:
        raise ValueError(f"Expected 100 offline trials, 50 left and 50 right; got {labels}/{counts}")
    return X, y


def zscore_fit(X):
    # Exact released arithmetic; input unit must be microvolts.
    Xd = X.astype(np.float64)
    return Xd.mean(axis=(0, 2), keepdims=True), Xd.std(axis=(0, 2), keepdims=True) + 1e-6


def zscore_apply(X, mean, sd):
    return ((X.astype(np.float64) - mean) / sd).astype(np.float32)


def subject_paths(data_dir, subject):
    if not 1 <= subject <= 54:
        raise ValueError("Lee subjects must be 1..54")
    stem = f"subject_{subject:02d}"
    folder = Path(data_dir)
    return dict(train=folder / f"{stem}_train.npz", test=folder / f"{stem}_test.npz",
                meta=folder / f"{stem}_meta.json")
