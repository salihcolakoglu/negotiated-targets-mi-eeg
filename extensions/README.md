# Completed additional studies: v2.2.0

This v2.2.0 release contains the completed additional studies and the W4 mechanism control described below. The version 2.1.0 DOI identifies the earlier archive only; it is not the DOI of these expanded results.

The original `src/`, `reference/stored_results.zip`, `reproduce.py` commands and legacy scientific records are retained. This extension adds three completed studies and a completed mechanism control, without combining their participants into one analysis:

| Study | Design | Retained test records |
|---|---|---:|
| `lee54` | 54 people, five starts, S0/W0/W1/W2/W3; first-day offline training, second-day offline testing | 1,350 |
| `lee_calibration80_20` | Separate exploratory retraining with 80 first-day training and 20 first-day calibration trials; paired raw/scaled probabilities | 1,350 |
| `shu_2x2` | Exploratory EEGNet/FBCNet × CE/Walsh, all DIRECT; 10 source and 15 target people; four later recording days | 1,320 |
| `lee_w4` | Same 54 Lee participants and five paired starts; exploratory ungated-anchoring control with 540 reused W1/W2 comparisons | 270 new |

SHU's total includes 1,200 neural and 120 classical test records. Its 400 neural fits include source controls and source models, which are not 400 independent participants. A separate completed 270-fit Lee W4 ungated anchored control reuses the 540 historical W1/W2 prediction files for paired comparisons; those reused files are not extra participants or additional fits.

## Verify retained predictions

From the repository root:

```bash
python -m pip install -r requirements.txt
python extensions/verify_retained.py --report outputs/extension_verification.json
```

The command needs NumPy and the standard library. It performs no EEG download, model loading, training or inference. It validates all archive member hashes and identities, recomputes native accuracy and probability metrics from 4,020 retained test files, and checks the Lee and SHU condition tables. It also verifies the 400 neural training budgets and transfer links, 50 classical fits, 330 reserved-Day-1 temperature inputs, 60 source validation records and 54 Lee classical predictions. It keeps the historical Lee class-probability NLL clipping distinct from the stable class-score NLL used in the later controls. It does not rerun inferential tests or optimize temperatures merely because their archived values are included.

`reference/new_studies_results.zip` contains test predictions, run/auxiliary indices, reserved calibration scores and scientific tables. It contains no EEG signals, checkpoints, optimization state, private journals or correspondence. The Lee54 archive stores only `P_test` and `y_test`; original training predictions and normalization arrays are omitted. Public JSON documents redact host and machine-specific input paths. These are attributed derivatives, with original source-file hashes retained in `SOURCE_PROVENANCE.json`.

## Verify the completed W4 mechanism control

```bash
python -m pip install -r extensions/requirements-verification.txt
python extensions/verify_w4.py --report outputs/w4_verification.json
```

This separate command uses NumPy and SciPy, verifies all 270 new prediction files and explicitly pairs
their prepared-input and source hashes to the 540 W1/W2 files already in the base
extension archive. It recomputes the native float64 Walsh decisions, historical
clipped NLL, ECE15, BCE, total Brier, confidence, 54 people's five-start means,
and exactly two new paired accuracy tests with two-test Holm correction.
The reported 95% intervals are pointwise, not multiplicity-adjusted intervals.
Saved training accuracy, target displacement and wall time are linked metadata;
they are not reproduced by this no-training command, and different checkpoint
overhead prevents a controlled training-speed comparison.

`W4_RESULTS_MANIFEST.json` is the authoritative completed-result addendum. The
unchanged `new_studies_results.zip` index and original source-only public template
retain their earlier pending snapshot notes; the original verifier consequently
continues to check its original three-study scope. They do not override the new
completed manifest. No new independent dataset validation is claimed.

## Scientific source snapshots and fresh training

`studies/*/source/` preserves the study-specific scientific source files with exact source hashes. These snapshots preserve the recorded model/filter/loss/update definitions. A historical campaign driver may still expect its original frozen workspace metadata and is **not** a new public training entry point. Do not substitute a redacted public protocol for its historical hash manifest.

The portable prepared-input controller supplies new local paths and prospective phase gates while retaining the original numerical kernels. See [PREPARED_INPUTS.md](PREPARED_INPUTS.md) for the exact data roles and boundaries.

```bash
python -m pip install -r extensions/requirements-training.txt
python extensions/train.py init --study lee54 --data-root /path/to/prepared_lee --work /path/to/new_lee_run --device cuda
```

The init command prints the new `PUBLIC_PROTOCOL.json` path and SHA256. Use those exact values in each stage:

```bash
python extensions/train.py validate --protocol /path/to/new_lee_run/PUBLIC_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train.py fit --protocol /path/to/new_lee_run/PUBLIC_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train.py freeze --protocol /path/to/new_lee_run/PUBLIC_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train.py test --protocol /path/to/new_lee_run/PUBLIC_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train.py analyze --protocol /path/to/new_lee_run/PUBLIC_PROTOCOL.json --protocol-sha256 PRINTED_SHA
```

Use `--study lee_calibration80_20` or `--study shu_2x2` and a different new work directory for the other studies. Lee54 and SHU support CPU or CUDA; the unchanged historical calibration driver requires CUDA for its memory telemetry and rejects `--device cpu` explicitly. The controller records exact software versions, avoids silently falling back to a different device, and retains phase/block checkpoints. Concurrent commands on one work directory are blocked. After an interrupted process, inspect the lock owner before manually recovering a stale lock; never delete a live lock.

The public wrapper's generated-input QA checks all three CLI initialization/validation/import paths in three working directories, closed test gates, input/code tampering, source write rejection and a tiny CPU kernel smoke test. It does **not** establish a new full training rerun or bitwise equivalence across hardware. Stored-result verification and fresh-training execution are separate claims. The completed W4 retained-result verification is separate from the generated-input execution QA; it does not assert a second full public training rerun.

## W4 mechanism control: source and fresh execution

The root-frozen W4 source is provided separately under `studies/lee_w4/source/`.
It adds ungated hard-label anchoring while retaining the original EEGNet,
float32 target-update arithmetic, seeds, 100 Day-1 trials and final 300 epochs.
The historical float64 Walsh metric decoder is preserved. W2−W4 isolates
correctness gating; W4−W1 compares anchored and cumulative updates. Their two
accuracy comparisons form a new exploratory Holm family. The original
five-comparison Lee family is retained.

A fresh W4 run requires a **completed public Lee54 work directory** with all
1,350 neural and 54 classical fits/tests and the same prepared input bytes.
It does not silently pair newly processed signals with historical predictions.
The reference work is read-only, and only its 540 W1/W2 records enter the new
paired comparison.

```bash
python extensions/train_w4.py init --reference-work /path/to/completed_lee_run --work /path/to/new_w4_run
python extensions/train_w4.py validate --protocol /path/to/new_w4_run/PUBLIC_W4_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train_w4.py fit --protocol /path/to/new_w4_run/PUBLIC_W4_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train_w4.py freeze --protocol /path/to/new_w4_run/PUBLIC_W4_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train_w4.py test --protocol /path/to/new_w4_run/PUBLIC_W4_PROTOCOL.json --protocol-sha256 PRINTED_SHA
python extensions/train_w4.py analyze --protocol /path/to/new_w4_run/PUBLIC_W4_PROTOCOL.json --protocol-sha256 PRINTED_SHA
```

This route requires CUDA for campaign fitting. It freezes all 270 new final
models and 54 first-day normalizers (864 artifacts) before Day-2 decoding;
historical comparator arrays wait until all 270 new tests (540 artifacts) pass.
The separate public protocol adapts path/provenance checks and does not
impersonate the root's historical workstation/IEEE-file manifest. W4 outcome
files are supplied separately in `reference/lee_w4_results.zip` after the actual
campaign and independent final numerical audit passed.

## Interpretation

The first Lee54 experiment was fixed before the team's first second-day evaluation. Later calibration, W4 and SHU analyses use datasets whose outcomes had already been inspected and are exploratory. Positive temperature scaling changes confidence but preserves the original class decision. The SHU FBCNet uses an explicitly adapted filter bank and variance implementation; its low performance is not a claim that the original published FBCNet recipe was reproduced and failed.

## Data and code attribution

- Lee/OpenBMI data: [10.5524/100542](https://doi.org/10.5524/100542), CC0-1.0; [Lee et al., GigaScience (2019)](https://doi.org/10.1093/gigascience/giz002).
- SHU: exact [Figshare version 1 MAT release](https://doi.org/10.6084/m9.figshare.19228725.v1), CC BY 4.0; [Ma et al., Scientific Data 9, 531 (2022)](https://doi.org/10.1038/s41597-022-01647-1). The paper, release and later EDF/NEMAR versions are distinct sources.
- FBCNet: [Mane et al., arXiv:2104.01233 (2021)](https://arxiv.org/abs/2104.01233); the [author implementation](https://github.com/ravikiran-mane/FBCNet) uses MIT. Its notice is preserved in `LICENSES/FBCNet-MIT.txt`.

The new software follows MIT. The authors' generated predictions and documentation use CC BY 4.0. Source labels retain their upstream data terms; this does not relicense the original recordings. The older BNCI datasets and their notices in the main repository remain unchanged.
