# Isolated Lee EEGNet extension

This directory is a new-data extension. The public v2.1.0 source and its retained results are not modified.

## Scientific behavior

- The released EEGNet-8,2 convolution layers, initializations, max-norm constraints, optimizer, losses and dropout are retained.
- S0 uses two class logits with cross-entropy. W0–W3 use the same first two 16-bit Walsh rows as the release and binary cross-entropy.
- W0 keeps fixed targets. W1 uses the cumulative released update. W2 restarts from the true codeword and updates only correctly decoded training trials. W3 moves toward the mean codeword without using model predictions.
- The zero-based phase index is retained: update rate is `0.03 * p` after training phase `p`. The update is used at the next phase. Phase 0 makes a zero-rate update; the last post-training update is not used in a later fit. `Y_used_last` records the target actually used to train the last phase.
- A full run is always 10 phases × 30 epochs, batch 16, Adam 0.001, dropout 0.5. There is no test-based early stopping, checkpoint selection or hyperparameter search.
- Training takes only Day 1 arrays. Day 2 prediction is a separate call after the fixed final fit. No phase-level test accuracy is calculated.
- Per-channel mean and standard deviation are fitted only on Day 1 over its trials and time samples. The released `std + 1e-6` arithmetic is retained, requiring microvolt input. Day 2 uses the same mean and standard deviation.
- CUDA execution uses float32, no automatic mixed precision, deterministic settings matching the release, and explicit synchronization for time measurements. CPU and CUDA are separate numerical environments; equivalence checks compare the extension against the release on the same device.

## Input contract

The data preparer supplies separate files for each participant:

```
subject_01_train.npz   X_train: float32 (100, 62, T), y_train: int64 (100,)
subject_01_test.npz    X_test: float32 (100, 62, T), y_test: int64 (100,)
subject_01_meta.json  subject: 1, data_units: "uV", plus preparation provenance
```

Labels are 0 = left-hand imagery and 1 = right-hand imagery; each day contains 50 trials per class. T is determined by the documented MNE epoch/resampling behavior; it is not forced to a guessed length. Channel order must match across days and only the 62 EEG channels are accepted. Separate train/test files allow the timing pilot to run without loading a test array.

## Timing pilot

```
python lee_train.py pilot --train-npz PREPARED/subject_01_train.npz --out TIMING_FULL --device cuda
```

The defaults perform one full-length fit each for W0, W1 and W2, seed 1, using only Day 1. To check technical execution before these fits, use a separate output folder and `--epochs-per-phase 1 --phases 2`. Pilot output is explicitly marked as timing, not scientific validation. It contains no test predictions or test metrics.

## Full validation

```
python lee_train.py run --data-dir PREPARED --out FINAL_RESULTS --subjects 1-54 --device cuda
```

Default conditions S0 W0 W1 W2 W3 and seeds 1 2 3 4 5 produce 1,350 fits. Use `--subjects`, `--seeds`, `--conditions` and `--max-jobs` only for scheduling fixed jobs; the full-run training budget cannot be changed through the CLI. Review the frozen protocol before opening final test performance. This adapter does not resolve any outstanding prior-use confirmation by itself.

## Recovery and records

Each fit writes an atomic checkpoint after every phase, including model, Adam state, data-permutation generator, all random states and current/last-used targets. Repeating the same command resumes from completed phases or skips completed fits only if code, input hashes, software versions and run settings agree. A different code or data identity requires a separate output folder.

`RUNNING.lock` rejects concurrent writers. After an interrupted process is no longer alive, `--recover-stale-lock` explicitly checks its local PID before removing the stale lock. Never remove a live runner's lock.

Outputs include per-fit JSON records, hashed prediction NPZ files, checkpoints, an environment receipt, a process log and an invocation progress summary. Scientific metrics remain in per-fit files; the process log only exposes phase progress and elapsed time. CUDA peak allocated and reserved memory are recorded for each model. Input file integrity hashes are checked without decoding Day 2 until its final evaluation.

## Synthetic verification

```
python verify_adapter.py --device cpu --report adapter_verification_cpu.json
python verify_adapter.py --device cuda --report adapter_verification_cuda.json
```

These checks use artificial arrays only. They compare all five conditions against the original released implementation, check every target-update phase, and confirm that interruption/resume reproduces identical model parameters and outputs. The test fixture includes synthetic test arrays to verify endpoint equivalence; it never opens Lee Day 2 data. CPU verification passed on PyTorch 2.5.1+cpu. GPU verification must be run in the installed CUDA environment before claiming CUDA validation.
