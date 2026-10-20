# Prepared input contract

The fresh-training entry point is `extensions/train.py`. It accepts locally
prepared, separated-day numerical NPZ files. It does not download EEG or claim
one-command reproduction from publisher files. Signals are not redistributed.
Use a data directory outside the new work directory. JSON paths are local to
that new run; historical machine paths are never required.

## Lee2019 / OpenBMI

For subjects 01–54 provide:

| File | Required arrays |
|---|---|
| `subject_XX_train.npz` | `X_train`: float32 [100,62,512]; `y_train`: int64 [100] |
| `subject_XX_test.npz` | `X_test`: float32 [100,62,512]; `y_test`: int64 [100] |
| `subject_XX_meta.json` | subject integer; `data_units="uV"`; `normalization="none"`; original processing provenance |

Each day contains 50 left-hand (0) and 50 right-hand (1) trials. Sampling is
128 Hz and units are microvolts. The complete source definition is
`studies/lee54/source/NPZ_CONTRACT.json`. The exact original preparation source
and its CLI help are supplied as `studies/lee54/source/prepare_lee2019.py`.
Publisher MD5 values and source files must be obtained from the attributed
release. The held-out day is Day 2's **EEG_MI_train** offline recording;
EEG_MI_test is an online feedback recording and is not substituted.

The Lee54 runner learns scaling using all 100 Day-1 trials. The separate
calibration runner applies its deterministic 80/20 split and fits scaling only
on the 80 fitting trials. Both globally freeze every final model before opening
Day-2 arrays. The 54 contextual CSP–LDA heads in Lee54 use both ends of the CSP
spectrum, six features, and default SVD LDA. They are not the older three-feature
single-ended reference.

## SHU architecture/output comparison

`studies/shu_2x2/input_layout.public.json` specifies every relative filename and
its role. `splits.public.json` supplies the exact previously fixed original row
indices and label mapping. Do not reselect trials or use another Figshare
version. Data arrays are float64 in the publisher-derived microvolt scale,
29 original EEG channels, 512 samples at 128 Hz. Labels and indices are int64.

| Role | NPZ members and shape |
|---|---|
| Day-1 `split` for all 25 people | `X_train` [50,29,512], `y_train` [50], `X_calibration` [20,29,512], `y_calibration` [20], `train_original_indices_zero_based` [50], `calibration_original_indices_zero_based` [20] |
| Day-1 `source70` for 10 source people | `X_fit` [70,29,512], `y_fit` [70], `fit_indices_zero_based` [70] |
| Days 2–5 for 15 target people | `X_test` [N,29,512], `y_test` [N], `source_indices_zero_based` [N] |

The exact source order is [21,6,10,18,19,25,8,7,15,12]; the target order is
[22,14,17,4,1,16,2,5,11,20,9,24,3,23,13]. Fit/calibration counts are 25+25 and
10+10. `N` follows the original event rows, including the 99-trial subject001
Day4 recording. No row is silently dropped. Later NPZs also require the JSON
sidecars named in the layout. All file byte hashes are recorded prospectively;
the original published event ZIP is included for independent row/label checks.

Prepared SHU signals must follow the original fixed FIR/resampling epoch
pipeline used to create these inputs. This release exposes the prepared-input
interface and historical source/configuration, not a newly validated raw MAT
acquisition/preparation command. Passing the shape checks alone does not prove
that an independently prepared signal matches that pipeline. The historical
prepared hashes in the layout support exact-input comparison when those files
are available.

The runner fits 80 source-person-control neural models, 20 main source models,
300 target neural models, 50 classical references, 330 temperatures and 60
source-person validation records. It freezes that complete 2,530-artifact union
before any of the 1,320 later-day tests. CNNs additionally see 630/700 source
trials; classical references see only the local 50. This is not equal information.

## Boundary and interpretation

File byte hashes may be read prospectively. Numerical test arrays are closed
until the complete fit manifest passes. The controller rejects changed inputs,
code and software identities; public data sources are read-only. New outputs
receive a **new public execution protocol**, not a historical protocol hash.
CPU and GPU executions need not be bitwise identical. Inspect all final analysis
audits before interpreting new results. This wrapper's synthetic QA covers
path/import/phase guards and tiny generated kernel smoke tests, not a new full
400/1,350-model rerun.
