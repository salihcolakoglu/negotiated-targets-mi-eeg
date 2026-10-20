#!/usr/bin/env python3
"""Separate exploratory held-out calibration diagnostic, not a paper rerun.

Stages (each requires a frozen protocol and its SHA256):
  validate: verify configuration/code/data hashes, without loading Day 2 arrays.
  pilot:    Day 1 only, subject 1, seed 1, five conditions; excluded from results.
  fit:      1,350 new models on 80 Day-1 trials; T on 20 held-out Day-1 trials.
  freeze:   verify all 1,350 fit artifacts and commit the immutable global manifest.
  evaluate: only after global freeze, open Day 2 and evaluate paired raw/scaled scores.

No command accepts a test-dependent model choice or a scientific-budget override.
Old Lee code is copied locally verbatim, verified by SHA256 and never modified.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ORIGINAL_PROTOCOL_SHA = "493378c9982feff6182216eb7406f68799e1435284cd80d39d91b258130fcab8"
DEPENDENCIES = {
    "lee_core.py": "5614c5fc08a1ed4e5f57596241d4dfafe5df5c027b8a17dce166108b8abd2638",
    "lee_data.py": "49b4326d390f9f5d4bf363e829a41f8b8902bf668033fab3439ac61ca4184740",
}
CONFIG = dict(
    subjects=list(range(1, 55)), conditions=["S0", "W0", "W1", "W2", "W3"], seeds=[1, 2, 3, 4, 5],
    split=dict(seed=20261018, algorithm="numpy default_rng SeedSequence([seed,subject]); class order 0,1; first10 permutation indices per class calibration, remaining40 fit; sorted trial indices", fit_per_class=40, calibration_per_class=10, shared_across_conditions_and_seeds=True),
    normalization="fit per-channel float64 mean/std+1e-6 on 80 fit trials only; apply to calibration and test",
    model=dict(architecture="EEGNet-8,2", lr=0.001, batch=16, dropout=0.5, phases=10, epochs_per_phase=30, epochs_total=300, target_rate=0.03, walsh_bits=16, model_selection="none; fixed final model"),
    calibration=dict(temperature_per="participant x condition x initialization seed", bounds=[0.05, 100.0], optimizer="scipy minimize_scalar bounded in log(T); xatol1e-10 maxiter500", candidates="T1, lower endpoint, upper endpoint, optimizer; min NLL; ties within1e-12 prefer first (T1)", objective="unclipped stable class NLL", score_mapping="legacy float32 network probabilities; Walsh bit clip1e-7 Bernoulli class log scores; S0 log softmax probabilities clip1e-12", original_decoder_unchanged=True),
    design=dict(scope="exploratory calibration diagnostic after prior Lee results were inspected", planned_fits=1350, independent_unit="participant after averaging five initialization seeds", independent_n=54, calibration_fit_data="20 held-out Day-1 trials", test="100 offline Day-2 trials; global model and T freeze before first test array read", primary_comparison="same newly refitted model before vs after class-score temperature scaling", no_interim_comparisons=True, original_100_trial_results_not_paired_control=True, csp_reference_refit=False),
    analysis=dict(primary="within-condition paired participant delta stable NLL (calibrated minus raw)", family=["S0:cal-raw", "W0:cal-raw", "W1:cal-raw", "W2:cal-raw", "W3:cal-raw"], multiplicity="Holm across exactly five conditions", test="two-sided paired t-test", alpha=0.05, intervals="pointwise unadjusted 95% paired-delta t confidence intervals", secondary="ECE15 and summed multiclass Brier descriptive", analysis_only_after_all_1350=True, compare_against_old_100_train_models_inferentially=False),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def pid_alive(pid):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return False
            raise OSError(error, "Cannot safely determine whether lock owner is alive")
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, __import__("ctypes").byref(exit_code)):
                raise __import__("ctypes").WinError(__import__("ctypes").get_last_error())
            return exit_code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def exclusive_lock(root, recover=False):
    """One lock for pilot, fit, freeze and evaluation, not merely per command."""
    path = root / "RUNNING.lock"
    token = dict(pid=os.getpid(), host=platform.node(), started_utc=now())
    if path.exists() and recover:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("host") != platform.node() or pid_alive(old["pid"]):
            raise RuntimeError("Refusing to recover a live or remote lock")
        recovery = root / "recovery" / f"stale-lock-{time.time_ns()}.json"
        recovery.parent.mkdir(parents=True, exist_ok=True)
        recovery.write_bytes(path.read_bytes())
        path.unlink()
    with path.open("x", encoding="utf-8") as f:
        json.dump(token, f)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) == token:
            path.unlink()


def verify_protocol(path, expected_sha, *, all_inputs=True):
    if sha(path) != expected_sha:
        raise RuntimeError("Protocol SHA256 mismatch")
    p = json.loads(path.read_text(encoding="utf-8-sig"))
    if p.get("schema") != "lee2019-heldout-calibration-v1" or p.get("status") != "frozen_before_new_fit":
        raise RuntimeError("Unsupported or unfrozen calibration protocol")
    if p.get("scientific_config") != CONFIG:
        raise RuntimeError("Scientific configuration differs from frozen runner contract")
    expected_code = {str(q.relative_to(STUDY)).replace("\\", "/") for q in HERE.glob("*.py")}
    if set(p.get("code_sha256", {})) != expected_code:
        raise RuntimeError("Protocol must bind exactly every local Python source")
    for rel, digest in p["code_sha256"].items():
        if not rel.startswith("code/") or ".." in Path(rel).parts or sha(STUDY / rel) != digest:
            raise RuntimeError(f"Changed calibration code: {rel}")
    original = Path(p["original_study"]["root"]).resolve()
    if p["original_study"]["protocol_sha256"] != ORIGINAL_PROTOCOL_SHA or sha(original / "DENEY_PROTOKOLU-V1.0.json") != ORIGINAL_PROTOCOL_SHA:
        raise RuntimeError("Original frozen protocol identity changed")
    for name, digest in DEPENDENCIES.items():
        if sha(HERE / name) != digest or sha(original / "code" / name) != digest:
            raise RuntimeError(f"Local dependency is not the exact original source: {name}")
    execution = p["execution"]
    if execution != dict(device="cuda", threads=2, parallel_gpu_workers=1):
        raise RuntimeError("Execution must be one CUDA worker with two CPU threads")
    data = Path(p["data"]["prepared_dir"]).resolve()
    if data != (original / "data" / "prepared").resolve():
        raise RuntimeError("Only the existing separated-day prepared source is permitted")
    expected_names = {f"subject_{s:02d}_{tail}" for s in range(1, 55) for tail in ("train.npz", "test.npz", "meta.json")}
    if set(p["data"]["prepared_sha256"]) != expected_names:
        raise RuntimeError("Protocol must bind all 108 NPZ files and 54 metadata files")
    if all_inputs:
        # Hash bytes for integrity; never decode a Day-2 array here.
        for name, digest in p["data"]["prepared_sha256"].items():
            if sha(data / name) != digest:
                raise RuntimeError(f"Prepared input changed: {name}")
    p["_protocol_sha256"] = expected_sha
    p["_path"] = str(path.resolve())
    return p


def verify_current(p):
    verify_protocol(Path(p["_path"]), p["_protocol_sha256"], all_inputs=False)


def runtime_setup(p):
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    import numpy as np
    import scipy
    from lee_core import setup_torch
    dev = setup_torch(2)
    if dev.type != "cuda":
        raise RuntimeError("The frozen design requires CUDA; refusing CPU fallback")
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), host=platform.node(), deterministic_algorithms=True, deterministic_warnings_only=True, mixed_precision=False, threads=2)
    return dev, env


def jobs(pilot=False):
    return [(s, c, seed) for s in ([1] if pilot else CONFIG["subjects"])
            for seed in ([1] if pilot else CONFIG["seeds"]) for c in CONFIG["conditions"]]


def job_key(subject, condition, seed):
    return f"subject_{subject:02d}__{condition}__seed{seed}"


def identity(p, subject, condition, seed, runtime, *, pilot=False):
    name = f"subject_{subject:02d}"
    return dict(protocol_sha256=p["_protocol_sha256"], subject=subject, condition=condition, seed=seed, pilot=pilot,
                input_sha256={tail: p["data"]["prepared_sha256"][name + tail] for tail in ("_train.npz", "_test.npz", "_meta.json")},
                runtime=runtime)


def paths(base, key):
    return dict(record=base / "records" / (key + ".json"), calibration=base / "calibration" / (key + ".npz"),
                checkpoint=base / "checkpoints" / (key + ".pt"), temperature=base / "temperatures" / (key + ".json"))


def validate_artifact_record(base, record_path, expected_identity=None):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "FIT_AND_T_FROZEN" or (expected_identity is not None and record.get("identity") != expected_identity):
        raise RuntimeError(f"Changed or incomplete fit identity: {record_path}")
    for artifact in record["artifacts"].values():
        rel = Path(artifact["path"])
        if rel.is_absolute() or ".." in rel.parts or sha(base / rel) != artifact["sha256"]:
            raise RuntimeError(f"Artifact integrity failed: {artifact['path']}")
    return record


def record_artifact(base, path):
    return dict(path=str(path.relative_to(base)).replace("\\", "/"), sha256=sha(path), bytes=path.stat().st_size)


def fit_models(p, *, pilot=False, recover=False):
    dev, runtime = runtime_setup(p)
    import numpy as np
    import torch
    from calibration_math import fixed_split, fit_temperature, legacy_class_scores
    from lee_core import build_model, set_seed, fit_training_only, predict
    from lee_data import load_part, zscore_fit, zscore_apply, code_rows, RULES
    base = STUDY / ("pilot" if pilot else "fits")
    base.mkdir(parents=True, exist_ok=True)
    atomic_json(base / f"environment-{time.time_ns()}.json", dict(runtime=runtime, created_utc=now(), pilot=pilot))
    data_dir = Path(p["data"]["prepared_dir"])
    start = time.perf_counter()
    complete, newly_fit = 0, 0
    with exclusive_lock(STUDY, recover):
        if not pilot and (STUDY / "FROZEN_FITS_MANIFEST.json").exists():
            raise RuntimeError("Global fit manifest is frozen; no more fitting is permitted")
        for subject, condition, seed in jobs(pilot):
            verify_current(p)
            key = job_key(subject, condition, seed)
            current_identity = identity(p, subject, condition, seed, runtime, pilot=pilot)
            dest = paths(base, key)
            if dest["record"].exists():
                validate_artifact_record(base, dest["record"], current_identity)
                complete += 1
                continue
            if (STUDY / "STOP_AFTER_CURRENT_FIT").exists():
                atomic_json(STUDY / "STATUS.json", dict(status="STOPPED_AT_FIT_BOUNDARY", stage="pilot" if pilot else "fit", completed=complete, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))
                return
            train_path = data_dir / f"subject_{subject:02d}_train.npz"
            meta_path = data_dir / f"subject_{subject:02d}_meta.json"
            for q in (train_path, meta_path):
                if sha(q) != p["data"]["prepared_sha256"][q.name]:
                    raise RuntimeError(f"Changed Day-1 input or metadata: {q}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("subject") != subject or meta.get("data_units") != "uV" or meta.get("normalization") != "none":
                raise ValueError("Unexpected prepared metadata")
            Xall, yall = load_part(train_path, "train")
            if Xall.shape != (100, 62, 512):
                raise ValueError("Prepared Day-1 shape must be exactly100 x62 x512")
            fit_idx, cal_idx = fixed_split(yall, subject)
            mean, sd = zscore_fit(Xall[fit_idx])
            Xfit, Xcal = zscore_apply(Xall[fit_idx], mean, sd), zscore_apply(Xall[cal_idx], mean, sd)
            C = code_rows()
            walsh = condition != "S0"
            set_seed(seed)
            model = build_model("EEGNet", 62, Xfit.shape[2], 16 if walsh else 2, 0.5).to(dev)
            resume = None
            if dest["checkpoint"].exists():
                saved = torch.load(dest["checkpoint"], map_location=dev, weights_only=False)
                if saved["identity"] != current_identity:
                    raise RuntimeError(f"Checkpoint identity mismatch: {key}")
                resume = saved["state"]
                resume["generator"] = resume["generator"].cpu()
                resume["rng"]["torch"] = resume["rng"]["torch"].cpu()
                if resume["rng"]["cuda"] is not None:
                    resume["rng"]["cuda"] = [v.cpu() for v in resume["rng"]["cuda"]]

            def checkpoint(state):
                dest["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
                temporary = dest["checkpoint"].with_suffix(".tmp.pt")
                torch.save(dict(identity=current_identity, state=state), temporary)
                os.replace(temporary, dest["checkpoint"])

            def progress(phase, phases, seconds):
                atomic_json(STUDY / "STATUS.json", dict(status="RUNNING_DAY1_ONLY", stage="pilot" if pilot else "fit", job=key, phase=phase, phases=phases, completed=complete, expected=len(jobs(pilot)), train_seconds=seconds, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))

            print(f"Fit {key}: 80 training, 20 calibration; no Day-2 arrays", flush=True)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            fit_record, _ = fit_training_only(model, Xfit, yall[fit_idx], walsh=walsh, rule=RULES[condition], C=C, lr=0.001, epochs_per_phase=30, batch=16, seed=seed, dev=dev, phases=10, resume=resume, checkpoint=checkpoint, progress=progress)
            model.eval()
            with torch.no_grad():
                logits_cal = model(torch.from_numpy(Xcal).to(dev)).float().cpu().numpy()
            Pcal = predict(model, torch.from_numpy(Xcal).to(dev), walsh)
            scores = legacy_class_scores(Pcal, walsh=walsh, codes=C)
            temperature = fit_temperature(scores, yall[cal_idx])
            temperature.update(identity=current_identity, fitted_utc=now(), n_calibration=20, calibration_day=1, no_test_array_loaded=True)
            atomic_npz(dest["calibration"], fit_indices=fit_idx, calibration_indices=cal_idx, y_calibration=yall[cal_idx], P_calibration=Pcal, logits_calibration=logits_cal, class_log_scores_calibration=scores, normalization_mean=mean, normalization_sd=sd)
            atomic_json(dest["temperature"], temperature)
            frozen = dict(status="FIT_AND_T_FROZEN", identity=current_identity, job=key, frozen_utc=now(), n_fit=80, n_calibration=20, epochs_completed=fit_record["epochs_completed"], train_seconds=fit_record["train_seconds"], wall_seconds=time.perf_counter() - t0, phase_seconds=fit_record["phase_seconds"], cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(dev)), cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(dev)), temperature=temperature["temperature"], split_sha256=canonical_hash(dict(fit=fit_idx.tolist(), calibration=cal_idx.tolist())), day2_arrays_loaded=False, pilot_excluded_from_study_results=pilot,
                          artifacts={name: record_artifact(base, dest[name]) for name in ("checkpoint", "calibration", "temperature")})
            atomic_json(dest["record"], frozen)
            complete += 1
            newly_fit += 1
            print(f"Completed {key}; count {complete}/{len(jobs(pilot))}; no test evaluation", flush=True)
            del model, resume, Xall, Xfit, Xcal
        atomic_json(base / "COMPLETION.json", dict(status="PILOT_DAY1_ONLY_COMPLETE" if pilot else "ALL_DAY1_FITS_AND_T_READY_FOR_GLOBAL_FREEZE", completed=complete, expected=len(jobs(pilot)), newly_fit=newly_fit, elapsed_seconds=time.perf_counter() - start, protocol_sha256=p["_protocol_sha256"], no_day2_array_loaded=True, finished_utc=now()))
        atomic_json(STUDY / "STATUS.json", dict(status="PILOT_COMPLETE" if pilot else "DAY1_FITS_COMPLETE", completed=complete, expected=len(jobs(pilot)), protocol_sha256=p["_protocol_sha256"], updated_utc=now()))


def freeze_models(p, *, recover=False):
    base = STUDY / "fits"
    target = STUDY / "FROZEN_FITS_MANIFEST.json"
    with exclusive_lock(STUDY, recover):
        verify_current(p)
        entries, splits, runtime = [], {}, None
        for subject, condition, seed in jobs():
            key = job_key(subject, condition, seed)
            q = paths(base, key)["record"]
            r = validate_artifact_record(base, q)
            ident = r["identity"]
            if ident["protocol_sha256"] != p["_protocol_sha256"] or ident["pilot"] or (ident["subject"], ident["condition"], ident["seed"]) != (subject, condition, seed) or r["day2_arrays_loaded"] or r["epochs_completed"] != 300:
                raise RuntimeError(f"Fit identity or budget failed: {key}")
            if subject in splits and splits[subject] != r["split_sha256"]:
                raise RuntimeError("Split differs across conditions or initialization seeds")
            splits[subject] = r["split_sha256"]
            if runtime is None:
                runtime = ident["runtime"]
            if runtime != ident["runtime"]:
                raise RuntimeError("Mixed execution environments")
            entries.append(dict(job=key, record=record_artifact(STUDY, q)))
        expected_records = {job_key(*j) + ".json" for j in jobs()}
        if {q.name for q in (base / "records").glob("*.json")} != expected_records:
            raise RuntimeError("Missing or unexpected fit record(s)")
        manifest = dict(status="ALL_MODELS_AND_T_FROZEN_BEFORE_DAY2_READ", protocol_sha256=p["_protocol_sha256"], count=1350, frozen_utc=now(), runtime=runtime, participant_split_sha256={str(k): v for k, v in splits.items()}, entries=entries)
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            # Idempotent only if all frozen content, except original timestamp, matches.
            manifest["frozen_utc"] = old["frozen_utc"]
            if manifest != old:
                raise RuntimeError("Existing global manifest differs; never overwrite it")
            digest = sha(target)
            hash_path = STUDY / "FROZEN_FITS_MANIFEST.sha256"
            if hash_path.exists() and hash_path.read_text(encoding="ascii").split()[0] != digest:
                raise RuntimeError("Existing global manifest checksum differs")
            if not hash_path.exists():
                hash_path.write_text(digest + "  FROZEN_FITS_MANIFEST.json\n", encoding="ascii")
            print("Global manifest already frozen and verified", flush=True)
            return
        with target.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        digest = sha(target)
        (STUDY / "FROZEN_FITS_MANIFEST.sha256").write_text(digest + "  FROZEN_FITS_MANIFEST.json\n", encoding="ascii")
        atomic_json(STUDY / "STATUS.json", dict(status="GLOBAL_MODELS_AND_T_FROZEN", count=1350, manifest_sha256=digest, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))
        print("All 1,350 models and T frozen; no Day-2 arrays loaded", flush=True)


def verify_global_manifest(p):
    path = STUDY / "FROZEN_FITS_MANIFEST.json"
    expected = (STUDY / "FROZEN_FITS_MANIFEST.sha256").read_text(encoding="ascii").split()[0]
    if sha(path) != expected:
        raise RuntimeError("Global manifest integrity failed")
    m = json.loads(path.read_text(encoding="utf-8"))
    if m["status"] != "ALL_MODELS_AND_T_FROZEN_BEFORE_DAY2_READ" or m["protocol_sha256"] != p["_protocol_sha256"] or m["count"] != 1350 or len(m["entries"]) != 1350:
        raise RuntimeError("Wrong or incomplete global fit manifest")
    expected_keys = {job_key(*j) for j in jobs()}
    if {e["job"] for e in m["entries"]} != expected_keys:
        raise RuntimeError("Global manifest job identities are incomplete or duplicated")
    for entry in m["entries"]:
        art = entry["record"]
        if art["path"] != "fits/records/" + entry["job"] + ".json":
            raise RuntimeError("Unexpected global manifest record path")
        path = STUDY / art["path"]
        if sha(path) != art["sha256"]:
            raise RuntimeError(f"Frozen fit record changed: {entry['job']}")
        validate_artifact_record(STUDY / "fits", path)
    return m, expected


def evaluate_models(p, *, recover=False):
    dev, runtime = runtime_setup(p)
    import numpy as np
    import torch
    from calibration_math import legacy_class_scores, probability_metrics
    from lee_core import build_model, predict, nearest
    from lee_data import load_part, zscore_apply, code_rows
    base, out = STUDY / "fits", STUDY / "test_results"
    data_dir = Path(p["data"]["prepared_dir"])
    with exclusive_lock(STUDY, recover):
        m, frozen_sha = verify_global_manifest(p)
        if runtime != m["runtime"]:
            raise RuntimeError("Evaluation environment differs from frozen fit environment")
        first_event = STUDY / "DAY2_EVALUATION_STARTED.json"
        if first_event.exists():
            first = json.loads(first_event.read_text(encoding="utf-8"))
            if first.get("global_manifest_sha256") != frozen_sha or first.get("protocol_sha256") != p["_protocol_sha256"]:
                raise RuntimeError("First Day-2 evaluation event has a different frozen identity")
            atomic_json(STUDY / "evaluation_events" / f"resume-{time.time_ns()}.json", dict(resumed_utc=now(), first_started_utc=first["started_utc"], global_manifest_sha256=frozen_sha, protocol_sha256=p["_protocol_sha256"]))
        else:
            with first_event.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(dict(started_utc=now(), global_manifest_sha256=frozen_sha, protocol_sha256=p["_protocol_sha256"], note="All 1350 fits and temperatures verified before first Day-2 array read"), f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        completed = 0
        for subject, condition, seed in jobs():
            verify_current(p)
            if sha(STUDY / "FROZEN_FITS_MANIFEST.json") != frozen_sha:
                raise RuntimeError("Global manifest changed during evaluation")
            key = job_key(subject, condition, seed)
            source = paths(base, key)
            fit_record = validate_artifact_record(base, source["record"])
            ident = dict(fit_identity=fit_record["identity"], global_manifest_sha256=frozen_sha, fit_record_sha256=sha(source["record"]))
            dest = out / "records" / (key + ".json")
            prediction = out / "predictions" / (key + ".npz")
            if dest.exists():
                r = json.loads(dest.read_text(encoding="utf-8"))
                if r["identity"] != ident or sha(prediction) != r["prediction_sha256"]:
                    raise RuntimeError(f"Changed completed evaluation: {key}")
                completed += 1
                continue
            if (STUDY / "STOP_AFTER_CURRENT_FIT").exists():
                atomic_json(STUDY / "STATUS.json", dict(status="STOPPED_AT_TEST_JOB_BOUNDARY", completed=completed, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))
                return
            temperature_record = json.loads(source["temperature"].read_text(encoding="utf-8"))
            if temperature_record["identity"] != fit_record["identity"] or temperature_record["temperature"] != fit_record["temperature"]:
                raise RuntimeError("Temperature identity mismatch")
            temperature = float(temperature_record["temperature"])
            with np.load(source["calibration"], allow_pickle=False) as a:
                mean, sd = a["normalization_mean"], a["normalization_sd"]
            saved = torch.load(source["checkpoint"], map_location=dev, weights_only=False)
            if saved["identity"] != fit_record["identity"] or saved["state"]["next_phase"] != 10:
                raise RuntimeError("Final model checkpoint mismatch")
            walsh, C = condition != "S0", code_rows()
            model = build_model("EEGNet", 62, 512, 16 if walsh else 2, 0.5).to(dev)
            model.load_state_dict(saved["state"]["model"])
            # This is the first allowed Day-2 array read. No training follows it.
            test_path = data_dir / f"subject_{subject:02d}_test.npz"
            if sha(test_path) != p["data"]["prepared_sha256"][test_path.name]:
                raise RuntimeError("Day-2 prepared file changed")
            Xtest, ytest = load_part(test_path, "test")
            if Xtest.shape != (100, 62, 512):
                raise ValueError("Prepared Day-2 shape must be exactly100 x62 x512")
            Xt = torch.from_numpy(zscore_apply(Xtest, mean, sd)).to(dev)
            model.eval()
            with torch.no_grad():
                logits = model(Xt).float().cpu().numpy()
            P = predict(model, Xt, walsh)
            decoded = nearest(P, C).astype(np.int64) if walsh else P.argmax(1).astype(np.int64)
            scores = legacy_class_scores(P, walsh=walsh, codes=C)
            raw, post_raw = probability_metrics(scores, ytest, 1.0, decoded)
            scaled, post_scaled = probability_metrics(scores, ytest, temperature, decoded)
            if not np.array_equal(post_raw.argmax(1), post_scaled.argmax(1)) or raw["accuracy_original_decoder"] != scaled["accuracy_original_decoder"]:
                raise RuntimeError("Temperature changed decisions")
            atomic_npz(prediction, y_test=ytest, P_network=P, logits_test=logits, class_log_scores_test=scores, posterior_raw=post_raw, posterior_calibrated=post_scaled, original_decoder_predictions=decoded, temperature=np.asarray(temperature))
            atomic_json(dest, dict(status="PAIRED_RAW_CALIBRATED_TEST_COMPLETE", identity=ident, job=key, n_test=100, temperature=temperature, raw=raw, calibrated=scaled, decision_invariance_verified=True, prediction_file=str(prediction.relative_to(out)).replace("\\", "/"), prediction_sha256=sha(prediction), completed_utc=now()))
            completed += 1
            atomic_json(STUDY / "STATUS.json", dict(status="RUNNING_FROZEN_DAY2_EVALUATION", completed=completed, expected=1350, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))
            print(f"Final paired evaluation complete: {completed}/1350; no interim comparison", flush=True)
            del model, saved, Xt, Xtest
        atomic_json(out / "COMPLETION.json", dict(status="ALL_PAIRED_CALIBRATION_EVALUATIONS_COMPLETE", count=completed, expected=1350, global_manifest_sha256=frozen_sha, protocol_sha256=p["_protocol_sha256"], completed_utc=now(), analysis_pending=True))
        atomic_json(STUDY / "STATUS.json", dict(status="ALL_EVALUATIONS_COMPLETE_ANALYSIS_PENDING", completed=completed, expected=1350, protocol_sha256=p["_protocol_sha256"], updated_utc=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("show-config", "validate", "pilot", "fit", "freeze", "evaluate"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--recover-stale-lock", action="store_true")
    args = parser.parse_args()
    if args.command == "show-config":
        print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
        return
    if not args.protocol or not args.protocol_sha256:
        parser.error("A frozen protocol and its SHA256 are mandatory")
    p = verify_protocol(args.protocol, args.protocol_sha256)
    if args.command == "validate":
        print(json.dumps(dict(status="PROTOCOL_CODE_AND_INPUT_HASHES_VERIFIED", protocol_sha256=p["_protocol_sha256"], input_files=162, day2_arrays_loaded=False)))
    elif args.command in ("fit", "pilot"):
        fit_models(p, pilot=args.command == "pilot", recover=args.recover_stale_lock)
    elif args.command == "freeze":
        freeze_models(p, recover=args.recover_stale_lock)
    elif args.command == "evaluate":
        evaluate_models(p, recover=args.recover_stale_lock)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
