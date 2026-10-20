# Negotiated targets for motor imagery EEG

Code and retained results for **Walsh-coded and negotiated targets for motor imagery EEG decoding: a controlled evaluation of accuracy and confidence**, by Salih Çolakoğlu, Mustafa Onat and Nuri Korhan.

**Version 2.2.0 release scope.** The additional material accompanies an IEEE Access manuscript revision under preparation. This description does not imply journal acceptance or publication. This v2.2.0 release contains the completed 270-fit W4 control and its independent final audit; the older [v2.1.0 DOI](https://doi.org/10.5281/zenodo.23290274) identifies only that earlier release.

**Permanent archive for version 2.2.0:** [DOI 10.5281/zenodo.23504655](https://doi.org/10.5281/zenodo.23504655). This is the software/results archive DOI, not a journal manuscript DOI.

The archive keeps the two original, matched-budget experiments (615 CSP–CNN and 800 EEGNet fits in the same 32 people) and adds separate evidence:

| Material | Independent analysis unit | Scope |
|---|---|---|
| Original experiments | 32 people, not 1,415 independent observations | Fixed/cumulative/anchored/smoothed target rules and classical references |
| Lee2019 external-data evaluation | 54 people's five-start means | 1,350 EEGNet fits plus 54 CSP–LDA references; fixed before the team's first second-day evaluation |
| Lee2019 held-out calibration | The same 54 people, analyzed separately | 1,350 new models, 80 fitting and 20 calibration trials; exploratory |
| SHU architecture/output control | 15 target people, conditional on 10 source people | 400 neural fits across source controls, source models and targets; exploratory |
| Lee2019 W4 mechanism control | The same 54 people, analyzed separately | 270 completed ungated-anchoring fits, paired with 540 historical W1/W2 records; a separate exploratory two-contrast family and retained-result verifier |

The controlled comparisons distinguish accuracy from confidence. They do not establish general superiority, prevention of forgetting, or clinical readiness. Repeated models, recording days and reused participants are not pooled into a larger independent sample.

For the new studies, start with [the extension guide](extensions/README.md). Its portable training commands require explicitly prepared inputs with the documented boundaries; they are separate from the no-training verification of retained predictions. [Prepared input contracts](extensions/PREPARED_INPUTS.md) specify the exact roles and preprocessing.

Verify the additional retained results without EEG or model loading:

```bash
python -m pip install -r extensions/requirements-verification.txt
python extensions/verify_retained.py --report outputs/extension_verification.json
python extensions/verify_w4.py --report outputs/w4_verification.json
```

The original extension verifier uses NumPy and the standard library. The separate
W4 paired-test verifier also uses SciPy; its pinned verification dependencies are
kept separate from the original requirements and from training dependencies.

The following original commands remain available and their scientific code and result bytes are preserved.

## 1. Check the reported results

From this repository's root, in a Python 3.11 or 3.12 environment:

```bash
python -m pip install -r requirements.txt
python reproduce.py verify
```

This checks all 1,415 run records, participant summaries, eight paired comparisons and 800 saved EEGNet prediction files. It prints `PASS` and saves a JSON report under `outputs/`. It needs no GPU or raw EEG download and performs no training. The reference archive is about 15 MB and is extracted into a temporary directory automatically.

## 2. Prepare inputs and run Experiment 1

Use a fresh **Python 3.11** virtual environment for the training dependencies. The recorded experiments used Linux CPU execution. On Windows, use a short environment path (for example `C:\venvs\eeg`) to avoid installer path-length limits. After activating the environment:

```bash
python -m pip install -r requirements/csp.txt
python reproduce.py prepare-csp
python reproduce.py prepare-eegnet
python reproduce.py train-csp
python reproduce.py classical
```

Preparation downloads the public recordings through MOABB into `data/` and writes generated inputs under `work/`. Allow several GB of free space. `prepare-eegnet` verifies the chronological trial-label splits against the CSP preparation before producing the wider-band EEG inputs. It does not train a model.

`train-csp` runs the 615 fits. To check one full-length fit first, add `--max-runs 1`; repeating the command resumes completed work. The optional comparison to the archived seed-1 run is:

```bash
python reproduce.py train-csp --reference reference/csp_seed1_reference.json
```

Differences are reported. Add `--strict-reference` only when intentionally requiring equality within 1e-9 on a matching platform. `classical` fits the specified CSP–LDA reference without parameter tuning.

## 3. Run Experiment 2

Activate a separate Python 3.11 virtual environment for PyTorch, using the same repository and prepared `work/` inputs:

```bash
python -m pip install -r requirements/eegnet.txt --index-url https://download.pytorch.org/whl/cpu
python reproduce.py train-eegnet
```

This runs the 800 EEGNet fits on CPU, using two workers and one thread per worker as in the recorded run. Use `--max-jobs 1` for one full-length fit; `--workers` and `--threads` control CPU parallelism. Runs resume from verified completed outputs. A short pipeline check is available with `--smoke --out outputs/eegnet-smoke`; smoke outputs are not study results. The complete experiments can take hours, depending on the machine.

After both full experiments finish:

```bash
python reproduce.py summarize
```

This creates participant means, paired estimates, confidence intervals, Holm-adjusted p values and readable accuracy tables in `outputs/tables/`. It rejects incomplete runs and smoke results. Use `python reproduce.py COMMAND --help` for path and run-selection options.

## Protocol

| Dataset | Participants | Classes | Training → test |
|---|---:|---:|---|
| BNCI2014-001 (D1) | 9 | 4 | Session 1 → 2; 288/288 trials |
| BNCI2014-002 (D3) | 14 | 2 | First 80 → last 80 trials in one session |
| BNCI2014-004 (D4) | 9 | 2 | Session 1 → 5; 120/160 trials, except B08: 160/160 |

CSP uses the archived one-ended spatial-filter selection, producing 8/3/3 features. All Experiment 1 conditions use 200 epochs and dropout 0.1. EEGNet uses 4–38 Hz EEG at 128 Hz, training-only channel standardization, 300 epochs and dropout 0.5. Budgets match within each experiment. Targets are updated after a phase; the last update used for training has coefficient 0.24.

Accuracy uses nearest-codeword decoding for Walsh outputs. Probability scores use the separately specified Bernoulli-to-class mapping. Seeds are averaged within participants before paired tests; four contrasts are corrected separately per experiment. Saved-result verification reproduces the stored calculations; fresh training can differ across hardware and software versions.

## Files and attribution

- `src/csp/`: data preparation, CSP–CNN and CSP–LDA.
- `src/eegnet/`: EEGNet preparation and five training conditions.
- `src/verify_results.py`, `src/summarize.py`: checking and analysis.
- `reference/`: retained results/predictions and the optional CSP reference.

The original commands cover the two original matched-budget experiments and their specified classical reference. The later studies have separate commands, protocols and result archives under `extensions/`; their sample sizes and statistical families are not merged with the original experiments.

Raw EEG is obtained from the [BNCI database](https://bnci-horizon-2020.eu/database/data-sets). The three source datasets are licensed under **CC BY-ND 4.0** by the **Institute for Knowledge Discovery, Graz University of Technology**. See [dataset attribution and access](DATA_ATTRIBUTION.md) for the individual dataset descriptions and related publications. Raw EEG and filtered/epoched signal caches are not included in this repository; upstream data licenses remain unchanged. The implementation follows [EEGNet](https://doi.org/10.1088/1741-2552/aace8c), [Walsh-coded outputs](https://arxiv.org/abs/2104.00035) and [negotiated representations](https://ijnes.org/index.php/ijnes/article/view/957); implementation details are in the accompanying manuscript. AI tools assisted code preparation and checking; the authors are responsible for the work.

Software is distributed under **MIT**. The authors' original documentation, generated predictions, targets and numerical summaries use **CC BY 4.0**. The original BNCI source-label arrays retain **CC BY-ND 4.0**. Later Lee/OpenBMI labels retain **CC0-1.0**, and SHU labels retain **CC BY 4.0**. These notices do not relicense source recordings. See [Licensing](LICENSING.md) for the scope and notices. Earlier MIT permissions remain valid.

## How to cite

Use [`CITATION.cff`](CITATION.cff) for the software version actually used. The candidate metadata names version 2.2.0 without assigning the older release's DOI to it. The new DOI will be linked only after the release and its Zenodo record have been verified.

The historical [v2.1.0 software archive](https://doi.org/10.5281/zenodo.23290274) remains available with its original code, results and citation. Its DOI is a software DOI, not a journal article DOI.

The accompanying study is:

> Çolakoğlu, S., Onat, M., and Korhan, N. (2026). Walsh-coded and negotiated targets for motor imagery EEG decoding: a controlled evaluation of accuracy and confidence. Manuscript prepared for IEEE Access review.

Please cite the relevant original data sources in [dataset attribution and access](DATA_ATTRIBUTION.md). Academic citation and software change-description requests do not add conditions to MIT; attribution and modification notices apply to original non-code material within CC BY 4.0's scope.

The retained-result checks recompute saved predictions and numerical summaries. Generated-input CLI checks exercise initialization, validation, gates and small kernel smoke tests. They do not claim a fresh rerun of every full training experiment, a fresh raw-data download, or bitwise training equivalence on another platform. The extension guide documents these distinctions.
