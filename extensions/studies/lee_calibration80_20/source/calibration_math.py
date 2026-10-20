"""Held-out, class-score temperature scaling without changing the EEG decoder.

The primary score map intentionally retains the released float32 network
probabilities and clipped independent-Bernoulli Walsh class-score map.  Fitting
uses stable, *unclipped* class NLL; reporting also retains legacy NLL clipped at
1e-12.  Neither fit nor metric routines accept EEG input or access test files.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _labels(y, n):
    y = np.asarray(y)
    if y.dtype != np.int64 or y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected int64 binary labels with one entry per sample")
    return y


def validate_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite nonempty n x 2 class log scores")
    return scores


def legacy_class_scores(probabilities, *, walsh, codes):
    """Retain the original probability-to-class-score map, before temperature."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Network probabilities must be finite and in [0,1]")
    if walsh:
        c = np.asarray(codes, dtype=np.float64)
        if c.shape != (2, 16) or p.shape[1] != 16 or not np.isin(c, [0, 1]).all():
            raise ValueError("Expected the two released 16-bit binary code rows")
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        scores = np.log(clipped) @ c.T + np.log(1 - clipped) @ (1 - c.T)
    else:
        if p.shape[1] != 2 or not np.allclose(p.sum(1), 1, atol=2e-7, rtol=0):
            raise ValueError("Expected two-class softmax probabilities")
        scores = np.log(np.clip(p, 1e-12, 1.0))
    return validate_scores(scores)


def log_posterior(scores, temperature=1.0):
    scores = validate_scores(scores)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and strictly positive")
    scaled = scores / float(temperature)
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def stable_nll(scores, labels, temperature=1.0):
    lp = log_posterior(scores, temperature)
    y = _labels(labels, len(lp))
    return float(-lp[np.arange(len(lp)), y].mean())


def fit_temperature(scores, labels, bounds=(0.05, 100.0)):
    """Fit one positive scalar using only the held-out calibration labels.

    Deterministic ties within 1e-12 prefer T=1. Endpoints are considered
    explicitly. A failed optimizer is an error, not a silent T=1 fallback.
    """
    scores = validate_scores(scores)
    y = _labels(labels, len(scores))
    lo, hi = map(float, bounds)
    if not (0 < lo < 1 < hi) or not np.isfinite([lo, hi]).all():
        raise ValueError("Bounds must be finite and contain T=1 strictly")
    objective = lambda log_t: stable_nll(scores, y, float(np.exp(log_t)))
    fit = minimize_scalar(objective, bounds=(np.log(lo), np.log(hi)), method="bounded",
                          options={"xatol": 1e-10, "maxiter": 500})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Temperature optimizer failed: {fit.message}")
    candidates = [("identity", 1.0), ("lower", lo), ("upper", hi),
                  ("optimizer", float(np.exp(fit.x)))]
    values = [(name, t, stable_nll(scores, y, t)) for name, t in candidates]
    minimum = min(v[2] for v in values)
    chosen = next(v for v in values if v[2] <= minimum + 1e-12)
    name, temperature, nll_after = chosen
    nll_before = values[0][2]
    if not lo <= temperature <= hi or nll_after > nll_before + 1e-10:
        raise RuntimeError("Invalid or worse calibration objective")
    # This is an exact ordering property of positive scalar class-score scaling.
    if not np.array_equal(scores.argmax(1), log_posterior(scores, temperature).argmax(1)):
        raise RuntimeError("Temperature unexpectedly changed posterior decisions")
    return dict(temperature=temperature, temperature_bounds=[lo, hi], chosen_candidate=name,
                boundary=bool(np.isclose(temperature, lo, rtol=1e-7, atol=0) or
                              np.isclose(temperature, hi, rtol=1e-7, atol=0)),
                calibration_nll_unclipped_before=nll_before,
                calibration_nll_unclipped_after=nll_after,
                objective="unclipped class NLL using stable logsumexp; calibration only",
                optimizer="scipy.optimize.minimize_scalar bounded in log(T)",
                optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                optimizer_nfev=int(fit.nfev), optimizer_xatol=1e-10,
                candidates=[dict(name=n, temperature=t, nll_unclipped=v) for n, t, v in values])


def probability_metrics(scores, labels, temperature, decoded_predictions):
    """Class-probability metrics; retain the separate original EEG decisions."""
    lp = log_posterior(scores, temperature)
    prob = np.exp(lp)
    y = _labels(labels, len(prob))
    decoded = _labels(np.asarray(decoded_predictions, dtype=np.int64), len(prob))
    pred = prob.argmax(1).astype(np.int64)
    confidence = prob.max(1)
    correct = (pred == y).astype(np.float64)
    bins, ece = [], 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for i in range(15):
        mask = ((confidence >= edges[i]) if i == 0 else (confidence > edges[i])) & (confidence <= edges[i + 1])
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        bins.append(dict(lower=float(edges[i]), upper=float(edges[i + 1]), count=count,
                         posterior_accuracy=accuracy, mean_confidence=mean_confidence))
    onehot = np.eye(2, dtype=np.float64)[y]
    return dict(nll_legacy_clipped=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
                nll_unclipped=float(-lp[np.arange(len(y)), y].mean()), ece_15_equal_width=float(ece),
                brier_sum_over_classes=float(np.square(prob - onehot).sum(1).mean()),
                accuracy_original_decoder=float((decoded == y).mean()),
                accuracy_class_posterior=float((pred == y).mean()),
                decoder_posterior_disagreement_fraction=float((decoded != pred).mean()),
                mean_max_probability=float(confidence.mean()), reliability_bins=bins), prob


def fixed_split(labels, subject, split_seed=20261018):
    y = _labels(labels, len(labels))
    if len(y) != 100 or np.bincount(y, minlength=2).tolist() != [50, 50]:
        raise ValueError("Expected exactly 50 trials per class")
    rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), int(subject)]))
    fit, calibration = [], []
    for klass in (0, 1):
        order = rng.permutation(np.flatnonzero(y == klass))
        calibration.extend(order[:10].tolist())
        fit.extend(order[10:].tolist())
    fit = np.asarray(sorted(fit), dtype=np.int64)
    calibration = np.asarray(sorted(calibration), dtype=np.int64)
    if len(np.intersect1d(fit, calibration)) or len(np.union1d(fit, calibration)) != 100:
        raise RuntimeError("Split overlap or incomplete coverage")
    return fit, calibration
