# Negotiated targets for motor imagery EEG

Code and retained results for **Negotiated label targets for motor imagery EEG decoding: a controlled evaluation with CSP features and EEGNet**, by Salih Çolakoğlu, Mustafa Onat and Nuri Korhan. Code package **v2.0**, accompanying manuscript **v4.1**.

The study compares fixed, cumulative, label-anchored and smoothed training targets on three public datasets. Experiment 1 contains 615 CSP–CNN fits; Experiment 2 contains 800 EEGNet fits. The independent sample is 32 participants.

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

This release covers the two completed matched-budget experiments and the specified classical reference. Historical experiments described separately in the supplement are not additional training modes here.

Raw EEG is obtained from the [BNCI database](https://bnci-horizon-2020.eu/database/data-sets) and remains subject to its providers' terms. The implementation follows [EEGNet](https://doi.org/10.1088/1741-2552/aace8c), [Walsh-coded outputs](https://arxiv.org/abs/2104.00035) and [negotiated representations](https://ijnes.org/index.php/ijnes/article/view/957); implementation details are in the accompanying manuscript. AI tools assisted code preparation and checking; the authors are responsible for the work.

Code is distributed under the **MIT license**. See `LICENSE` and `CITATION.cff`.

Release checks on Python 3.12/Windows passed stored-result verification, short CPU training with TensorFlow and PyTorch, and all 32 CSP–LDA accuracy comparisons. These checks did not repeat the full 1,415 training fits or a fresh download of the EEG recordings.
