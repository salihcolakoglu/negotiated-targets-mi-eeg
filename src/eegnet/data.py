"""Dataset loading and train-only standardization for the EEGNet experiment."""
import glob
import math
import os
import numpy as np
DATASETS = {
    "D1": dict(prefix="D1_BCIIV2a", name="BCI IV-2a", n_subjects=9, n_classes=4),
    "D3": dict(prefix="D3_BNCI2014_002", name="BNCI2014-002", n_subjects=14, n_classes=2),
    "D4": dict(prefix="D4_BCIIV2b", name="BCI IV-2b", n_subjects=9, n_classes=2),
}
R_RATE, PHASES = 0.03, 10
_CACHE = {}
def create_walsh(size_of_walsh=16):
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


W16 = create_walsh(16)

def code_rows(n_cls):
    return W16[:n_cls].astype(np.float32)


def load_subject(veri_dir, dk, s):
    """X (n, C, T) float32 in microvolt, y 1..K, train/test masks of the paper's split, session labels."""
    key = (os.path.abspath(veri_dir), dk, s)
    if key in _CACHE:
        return _CACHE[key]
    pre = DATASETS[dk]["prefix"]
    parts = sorted(glob.glob(os.path.join(veri_dir, f"{pre}_S{s:02d}*.npz")))
    if not parts:
        raise FileNotFoundError(f"no data for {pre}_S{s:02d} in {veri_dir}")
    Xs, meta = [], None
    for p in parts:
        with np.load(p, allow_pickle=False) as z:
            Xs.append(z["X"])
            if meta is None:
                meta = {k: z[k] for k in z.files if k != "X"}
    X = np.concatenate(Xs, 0).astype(np.float32)
    y = meta["y"].astype(np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{pre}_S{s:02d}: {X.shape[0]} trials in X but {y.shape[0]} labels")
    d = dict(X=X, y=y, train=meta["train_mask"].astype(bool), test=meta["test_mask"].astype(bool),
             session=meta["session"])
    _CACHE[key] = d
    return d


def zscore_fit(X):
    Xd = X.astype(np.float64)
    m = Xd.mean(axis=(0, 2), keepdims=True)
    sd = Xd.std(axis=(0, 2), keepdims=True) + 1e-6
    return m, sd


def zscore_apply(X, m, sd):
    return ((X.astype(np.float64) - m) / sd).astype(np.float32)

