#!/usr/bin/env python3
"""Synthetic scientific-regression checks; never opens Lee Day-2 data."""
import argparse
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import lee_core as new
import lee_data as data


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-dir", type=Path, default=Path(__file__).resolve().parents[2] /
                        "GITHUB_ZENODO_YAYIN_v2.1.0_2026-10-11/package/negotiated-targets-mi-eeg-v2.1.0/src/eegnet")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    # Absolute reference path supplied above; all originals are read-only.
    old_data = load_module("retained_data", args.released_dir / "data.py")
    previous = sys.modules.get("data")
    sys.modules["data"] = old_data
    try:
        old = load_module("retained_core", args.released_dir / "core.py")
    finally:
        if previous is None:
            del sys.modules["data"]
        else:
            sys.modules["data"] = previous
    dev = new.setup_torch(1)
    if args.device == "cpu":
        dev = torch.device("cpu")
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA regression requested but unavailable")
    C = data.code_rows()
    assert np.array_equal(C, old_data.code_rows(2))
    rng = np.random.default_rng(1729)
    Xtr = rng.normal(size=(8, 3, 128)).astype(np.float32)
    Xtest = rng.normal(size=(4, 3, 128)).astype(np.float32)
    ytr = np.array([0, 1] * 4, dtype=np.int64)
    ytest = np.array([0, 1] * 2, dtype=np.int64)
    old.PHASES = 3
    comparisons = []
    for condition in data.CONDITIONS:
        walsh = condition != "S0"
        n_out = 16 if walsh else 2
        old.set_seed(3)
        old_model = old.EEGNet(3, 128, n_out).to(dev)
        old_record, old_arrays = old.fit_phases(old_model, Xtr, ytr, Xtest, ytest,
                                               walsh=walsh, rule=data.RULES[condition], C=C,
                                               lr=0.001, epochs_per_phase=1, batch=4,
                                               seed=3, dev=dev)
        new.set_seed(3)
        new_model = new.EEGNet(3, 128, n_out).to(dev)
        new_record, new_arrays = new.fit_training_only(new_model, Xtr, ytr,
                                                     walsh=walsh, rule=data.RULES[condition], C=C,
                                                     lr=0.001, epochs_per_phase=1, batch=4,
                                                     seed=3, dev=dev, phases=3)
        final_record, final_arrays = new.evaluate_final(new_model, Xtest, ytest, walsh=walsh, C=C, dev=dev)
        for name, value in old_model.state_dict().items():
            assert torch.equal(value, new_model.state_dict()[name]), (condition, name)
        for name in ("P_train", "y_train"):
            assert np.array_equal(old_arrays[name], new_arrays[name]), (condition, name)
        for name in ("P_test", "y_test"):
            assert np.array_equal(old_arrays[name], final_arrays[name]), (condition, name)
        if walsh:
            assert np.array_equal(old_arrays["Y_used_last"], new_arrays["Y_used_last"])
            assert old_record["used_target_dist_by_phase"] == new_record["used_target_dist_by_phase"]
        for name in ("acc_test", "nll_test", "ece_test", "bce_test"):
            assert old_record[name] == final_record[name], (condition, name)
        assert new_record["phase_test_metrics_computed"] is False
        comparisons.append(dict(condition=condition, parameter_and_output_equality="bit_exact"))

    # Check all ten target updates, including the last update never used to fit.
    Y = C[ytr].copy()
    for rule in ("hard", "neg_paper", "neg_anchor_gate", "smooth_class"):
        actual = Y.copy()
        expected = Y.copy()
        for p in range(10):
            P = rng.uniform(0.05, 0.95, size=Y.shape).astype(np.float32)
            a = 0.03 * p
            if rule == "neg_paper":
                expected = (expected - (expected - P) * 0.03 * p).astype(np.float32)
            elif rule == "neg_anchor_gate":
                gate = old.nearest(P, C) == ytr
                expected = Y.copy()
                expected[gate] = ((1 - a) * Y[gate] + a * P[gate]).astype(np.float32)
            elif rule == "smooth_class":
                expected = ((1 - a) * Y + a * C.mean(0, keepdims=True).astype(np.float32)).astype(np.float32)
            actual, _ = new.update_targets(actual, Y, P, ytr, C, p, rule)
            assert np.array_equal(actual, expected), (rule, p)

    # Interrupted fit must resume with identical parameters, targets and outputs.
    options = dict(walsh=True, rule="neg_paper", C=C, lr=0.001, epochs_per_phase=1,
                   batch=4, seed=2, dev=dev, phases=4)
    new.set_seed(2)
    uninterrupted = new.EEGNet(3, 128, 16).to(dev)
    _, baseline = new.fit_training_only(uninterrupted, Xtr, ytr, **options)
    new.set_seed(2)
    partial = new.EEGNet(3, 128, 16).to(dev)
    saved = {}

    class PilotInterruption(Exception):
        pass

    def stop_after_phase_two(state):
        if state["next_phase"] == 2:
            saved.update(copy.deepcopy(state))
            raise PilotInterruption()

    try:
        new.fit_training_only(partial, Xtr, ytr, **options, checkpoint=stop_after_phase_two)
    except PilotInterruption:
        pass
    new.set_seed(999)
    resumed = new.EEGNet(3, 128, 16).to(dev)
    _, resumed_arrays = new.fit_training_only(resumed, Xtr, ytr, **options, resume=saved)
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    assert np.array_equal(baseline["P_train"], resumed_arrays["P_train"])
    assert np.array_equal(baseline["Y_used_last"], resumed_arrays["Y_used_last"])

    # Training normalization is exactly the archived function, independent of test.
    for expected, actual in zip(old_data.zscore_fit(Xtr), data.zscore_fit(Xtr)):
        assert np.array_equal(expected, actual)
    mean, sd = data.zscore_fit(Xtr)
    assert np.array_equal(old_data.zscore_apply(Xtr, mean, sd), data.zscore_apply(Xtr, mean, sd))
    signature = str(inspect.signature(new.fit_training_only))
    assert "Xte" not in signature and "yte" not in signature
    report = dict(status="PASS", device=str(dev), torch=torch.__version__, numpy=np.__version__,
                  released_source_sha256={name: data.sha(args.released_dir / name) for name in ("core.py", "data.py")},
                  conditions=comparisons, all_ten_update_rules="bit_exact",
                  checkpoint_resume="bit_exact", train_only_normalization="bit_exact",
                  phase_test_metrics_removed=True, lee_day2_data_opened=False,
                  scope="Synthetic regression only; not a scientific Lee result")
    if args.report:
        data.atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
