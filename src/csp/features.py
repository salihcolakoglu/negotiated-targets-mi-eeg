"""CSP functions preserved from REPRO_v1.0/scripts/T05_csp_feature_extraction.py.

Original project: Nuri Korhan and coauthors. Packaging and corrected description:
GPT-6 Astra / Codex, 2026-09-25. Arithmetic and the unusual filter selection are
unchanged. This is the paper's one-ended CSP, not a conventional two-ended CSP.
"""
import numpy as np


def bandpass_filter(data, low=8.0, high=30.0, fs=250.0, order=5):
    """Apply the original 8--30 Hz filter after MOABB's 8--32 Hz preprocessing."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[trial, ch, :] = filtfilt(b, a, data[trial, ch, :])
    return filtered


def compute_csp(x_train, y_train, x_test, n_components=6):
    """Fit spatial filters using training data only; return log-variance features."""
    classes = np.unique(y_train)
    if len(classes) != 2:
        return compute_csp_multiclass(x_train, y_train, x_test, n_components)
    c1, c2 = classes
    x1 = x_train[y_train == c1]
    x2 = x_train[y_train == c2]
    cov1 = np.mean([np.cov(trial) for trial in x1], axis=0)
    cov2 = np.mean([np.cov(trial) for trial in x2], axis=0)
    cov_composite = cov1 + cov2
    eigvals, eigvecs = np.linalg.eigh(cov_composite)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals, 1e-10)
    P = np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T)
    S1 = np.dot(np.dot(P, cov1), P.T)
    eig_vals_s1, B = np.linalg.eigh(S1)
    idx = np.argsort(eig_vals_s1)[::-1]
    B = B[:, idx]
    # Preserve the exact archived expression: the second slice is empty.
    # It selects indices 0..n_half-1 only, NOT first and last n_half filters.
    # Thus the study produces 8 features for D1 and 3 each for D3 and D4.
    n_half = n_components // 2
    W = np.dot(B[:, np.r_[:n_half, -n_half:]].T, P)
    x_train_csp = np.array([np.log(np.var(np.dot(W, trial), axis=1)) for trial in x_train])
    x_test_csp = np.array([np.log(np.var(np.dot(W, trial), axis=1)) for trial in x_test])
    return x_train_csp, x_test_csp


def compute_csp_multiclass(x_train, y_train, x_test, n_components=6):
    """Original one-vs-rest CSP, including its reported variance fallback."""
    classes = np.unique(y_train)
    n_per_class = max(2, n_components // len(classes))
    all_train_features = []
    all_test_features = []
    for cls in classes:
        y_binary = (y_train == cls).astype(int)
        x_cls = x_train.copy()
        y_cls = y_binary + 1
        try:
            feat_train, feat_test = compute_csp(x_cls, y_cls, x_test, n_per_class)
            all_train_features.append(feat_train)
            all_test_features.append(feat_test)
        except Exception as e:
            print(f"  CSP failed for class {cls}: {e}")
            feat_train = np.log(np.var(x_train, axis=2) + 1e-10)[:, :n_per_class]
            feat_test = np.log(np.var(x_test, axis=2) + 1e-10)[:, :n_per_class]
            all_train_features.append(feat_train)
            all_test_features.append(feat_test)
    return np.hstack(all_train_features), np.hstack(all_test_features)
