#!/usr/bin/env python3
"""Lee Day-1-only timing pilot, then fixed-budget final Day-2 validation.

Examples:
  python lee_train.py pilot --train-npz DATA/subject_01_train.npz --out TIMING --device cuda
  python lee_train.py run --data-dir DATA --out RESULTS --subjects 1-54 --device cuda

Pilot accepts no test file. Run has no budget override and evaluates only the
last fixed model. Completed outputs and phase checkpoints are resumed only
when code, data, parameters and software identity all match.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def parse_subjects(value):
    result = []
    for part in value.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    if not result or len(set(result)) != len(result) or any(s < 1 or s > 54 for s in result):
        raise argparse.ArgumentTypeError("Use unique Lee subject numbers 1..54")
    return result


@contextmanager
def output_lock(out, recover):
    path = out / "RUNNING.lock"
    if path.exists() and recover:
        old = json.loads(path.read_text(encoding="utf-8"))
        # Recovery is explicit, but still reject a demonstrably live local PID.
        if old.get("host") == platform.node():
            if not pid_alive(old["pid"]):
                path.unlink()
            else:
                raise RuntimeError(f"Runner PID {old['pid']} is still alive")
        else:
            raise RuntimeError("Cannot automatically recover another computer's lock")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), host=platform.node(),
                       started_utc=datetime.now(timezone.utc).isoformat()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def pid_alive(pid):
    """Read process existence; Windows os.kill(pid, 0) is deliberately avoided."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # Query limited information only.
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: nonexistent PID.
                return False
            raise OSError(error, "Could not safely establish whether the lock owner is alive")
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "run"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--recover-stale-lock", action="store_true")
        p.add_argument("--conditions", nargs="+", choices=("S0", "W0", "W1", "W2", "W3"),
                       default=["W0", "W1", "W2"] if name == "pilot" else ["S0", "W0", "W1", "W2", "W3"])
    p = commands.choices["pilot"]
    p.add_argument("--train-npz", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=range(1, 6), default=1)
    p.add_argument("--epochs-per-phase", type=int, default=30)
    p.add_argument("--phases", type=int, default=10)
    p = commands.choices["run"]
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--subjects", type=parse_subjects, default=list(range(1, 55)))
    p.add_argument("--seeds", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument("--max-jobs", type=int, default=0, help="Limit newly started fits; 0 completes all selected fits")
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.conditions)) != len(args.conditions):
        parser.error("Use positive threads and unique conditions")
    if args.command == "pilot" and not (1 <= args.epochs_per_phase <= 30 and 1 <= args.phases <= 10):
        parser.error("Pilot budget must be 1..30 epochs/phase and 1..10 phases")
    if args.command == "run" and (args.max_jobs < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use nonnegative max-jobs and unique seeds")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from lee_core import build_model, set_seed, setup_torch, fit_training_only, evaluate_final
    from lee_data import PARAMS, RULES, atomic_json, code_rows, load_part, sha, subject_paths, zscore_fit, zscore_apply
    dev = setup_torch(args.threads)
    if str(dev) != args.device:
        parser.error(f"Requested {args.device}, available device is {dev}; no silent fallback")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = dict(PARAMS)
    if args.command == "pilot":
        params.update(phases=args.phases, epochs_per_phase=args.epochs_per_phase)
        if not args.train_npz.name.endswith("_train.npz"):
            parser.error("Pilot requires an explicitly separate *_train.npz file")
        jobs = [(None, condition, args.seed) for condition in args.conditions]
    else:
        jobs = [(subject, condition, seed) for subject in args.subjects
                for seed in args.seeds for condition in args.conditions]
    code_identity = {name: sha(HERE / name) for name in ("lee_core.py", "lee_data.py", "lee_train.py")}
    common = dict(command=args.command, code_sha256=code_identity, parameters=params,
                  device=args.device, threads=args.threads,
                  python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                  cuda=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                  deterministic_algorithms=True, deterministic_warnings_only=True, mixed_precision=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json(out / f"environment_{stamp}.json", dict(common, host=platform.node(), selected_jobs=len(jobs)))
    log = (out / "process.jsonl").open("a", encoding="utf-8", buffering=1)

    def event(kind, **fields):
        record = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=kind, **fields)
        log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        log.flush()

    completed, newly_run = [], 0
    with output_lock(out, args.recover_stale_lock):
        for subject, condition, seed in jobs:
            paths = dict(train=args.train_npz.resolve()) if args.command == "pilot" else subject_paths(args.data_dir.resolve(), subject)
            # Hashing Day 2 checks file integrity but does not parse labels or predict.
            data_hash = {name: sha(path) for name, path in paths.items()}
            key = f"{paths['train'].stem}__{condition}__seed{seed}"
            identity = dict(common, dataset="Lee2019_MI", subject=subject, condition=condition,
                            seed=seed, data_sha256=data_hash)
            result_path = out / "records" / f"{key}.json"
            prediction_path = out / "predictions" / f"{key}.npz"
            checkpoint_path = out / "checkpoints" / f"{key}.pt"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["identity"] != identity or sha(prediction_path) != result["prediction_sha256"]:
                    raise RuntimeError(f"Changed identity or predictions for completed {key}; use a new output folder")
                completed.append(result)
                print(f"Already complete: {key}", flush=True)
                continue
            if args.command == "run" and args.max_jobs and newly_run >= args.max_jobs:
                break
            event("fit_started", job=key)
            print(f"Starting {key}: {params['phases']} x {params['epochs_per_phase']} epochs on {dev}", flush=True)
            start = time.perf_counter()
            Xraw, ytr = load_part(paths["train"], "train")
            if args.command == "run":
                meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
                if meta.get("data_units", meta.get("units")) != "uV":
                    raise ValueError("Prepared input must explicitly be in microvolts (uV)")
                if meta.get("subject") != subject:
                    raise ValueError("Subject metadata does not match the requested subject")
            mean, sd = zscore_fit(Xraw)
            Xtr = zscore_apply(Xraw, mean, sd)
            C = code_rows()
            walsh = condition != "S0"
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            set_seed(seed)
            model = build_model("EEGNet", Xtr.shape[1], Xtr.shape[2], 16 if walsh else 2, params["dropout"]).to(dev)
            resume = None
            if checkpoint_path.exists():
                # Only checkpoints created in this user-selected local output are loaded.
                saved = torch.load(checkpoint_path, map_location=dev, weights_only=False)
                if saved["identity"] != identity:
                    raise RuntimeError(f"Checkpoint identity changed for {key}; use a new output folder")
                resume = saved["state"]
                # CPU RNG states must stay CPU even when tensor weights map to CUDA.
                resume["generator"] = resume["generator"].cpu()
                resume["rng"]["torch"] = resume["rng"]["torch"].cpu()
                if resume["rng"]["cuda"] is not None:
                    resume["rng"]["cuda"] = [state.cpu() for state in resume["rng"]["cuda"]]
                event("fit_resumed", job=key, next_phase=resume["next_phase"])

            def checkpoint(state):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint_path.with_suffix(".tmp.pt")
                torch.save(dict(identity=identity, state=state), temporary)
                os.replace(temporary, checkpoint_path)

            def progress(phase, phases, seconds):
                event("phase_complete", job=key, phase=phase, phases=phases, train_seconds=seconds)
                print(f"{key}: phase {phase}/{phases}; {seconds:.2f}s; no test evaluation", flush=True)

            record, arrays = fit_training_only(model, Xtr, ytr, walsh=walsh, rule=RULES[condition], C=C,
                                               lr=params["lr"], epochs_per_phase=params["epochs_per_phase"],
                                               batch=params["batch"], seed=seed, dev=dev, phases=params["phases"],
                                               resume=resume, checkpoint=checkpoint, progress=progress)
            arrays.update(normalization_mean=mean, normalization_sd=sd)
            if args.command == "run":
                # Day 2 is first decoded here, after fixed training is finished.
                Xtest_raw, ytest = load_part(paths["test"], "test")
                if Xtest_raw.shape[1:] != Xraw.shape[1:]:
                    raise ValueError("Day 1 and Day 2 channel/time dimensions differ")
                final, test_arrays = evaluate_final(model, zscore_apply(Xtest_raw, mean, sd), ytest,
                                                   walsh=walsh, C=C, dev=dev)
                record.update(final, n_test=len(ytest))
                arrays.update(test_arrays)
                event("final_test_complete", job=key)
            else:
                record.update(purpose="training_only_timing_pilot_not_validation", final_test_evaluations=0)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_name(prediction_path.name + ".tmp.npz")
            np.savez(temporary, **arrays)
            os.replace(temporary, prediction_path)
            record.update(identity=identity, job=key, n_train=len(ytr), input_shape=list(Xraw.shape),
                          wall_seconds=time.perf_counter() - start,
                          cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
                          cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else None,
                          prediction_file=str(prediction_path.relative_to(out)).replace("\\", "/"),
                          prediction_sha256=sha(prediction_path),
                          completed_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(result_path, record)
            completed.append(record)
            newly_run += 1
            event("fit_complete", job=key, wall_seconds=record["wall_seconds"])
            print(f"Completed {key}: {record['wall_seconds']:.2f}s", flush=True)
            del model, resume
        summary = dict(command=args.command, selected_jobs=len(jobs), completed_selected_jobs=len(completed),
                       new_jobs_this_invocation=newly_run, all_selected_complete=len(completed) == len(jobs),
                       elapsed_training_seconds=sum(r["train_seconds"] for r in completed),
                       records=[r["job"] for r in completed],
                       test_performance_computed=args.command == "run" and bool(completed))
        atomic_json(out / "progress.json", summary)
        event("invocation_complete", **summary)
    log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
