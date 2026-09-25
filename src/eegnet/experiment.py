"""The five EEGNet conditions, five seeds, and 32 participants (800 runs)."""
import os
import numpy as np
from data import DATASETS, code_rows, load_subject, zscore_fit, zscore_apply
from core import build_model, set_seed, fit_phases
SMOKE = os.environ.get("NEGREP_SMOKE", "0") == "1"
SEEDS = [1, 2, 3, 4, 5]
D02_CONDS = ["S0", "W0", "W1", "W2", "W3"]
D02_RULE = {"S0": "hard", "W0": "hard", "W1": "neg_paper", "W2": "neg_anchor_gate", "W3": "smooth_class"}
D02 = dict(lr=1e-3, epochs=30, batch=16, dropout=0.5)

def _ep(n):
    return 1 if SMOKE else n


def _split(d):
    Xtr, Xte = d["X"][d["train"]], d["X"][d["test"]]
    ytr0, yte0 = d["y"][d["train"]] - 1, d["y"][d["test"]] - 1
    return Xtr, Xte, ytr0, yte0


def run_d02(veri, model_name, dk, s, cond, seed, dev):
    K = DATASETS[dk]["n_classes"]
    C = code_rows(K)
    Xtr, Xte, ytr0, yte0 = _split(load_subject(veri, dk, s))
    m, sd = zscore_fit(Xtr)
    Xtr, Xte = zscore_apply(Xtr, m, sd), zscore_apply(Xte, m, sd)
    walsh = cond != "S0"
    set_seed(seed)
    model = build_model(model_name, Xtr.shape[1], Xtr.shape[2], 16 if walsh else K, D02["dropout"]).to(dev)
    rec, arr = fit_phases(model, Xtr, ytr0, Xte, yte0, walsh=walsh, rule=D02_RULE[cond], C=C, lr=D02["lr"],
                          epochs_per_phase=_ep(D02["epochs"]), batch=D02["batch"], seed=seed, dev=dev)
    rec.update(experiment="D02", model=model_name, dataset=dk, subject=s, condition=cond, seed=seed,
               rule=D02_RULE[cond], n_train=int(len(ytr0)), n_test=int(len(yte0)))
    return [(rec, arr)]


def rec_key(r):
    return f"{r['experiment']}|{r['model']}|{r['dataset']}|{r['subject']}|{r['condition']}|{r['seed']}"


def job_list():
    subjects = [(dk, s) for dk, di in DATASETS.items() for s in range(1, di["n_subjects"] + 1)]
    return [("D02", "EEGNet", dk, s, cond, seed) for seed in SEEDS for cond in D02_CONDS for dk, s in subjects]


def run_job(job, data_dir, device):
    kind, model, dk, subject, condition, seed = job
    if kind != "D02" or model != "EEGNet":
        raise ValueError(job)
    return run_d02(data_dir, model, dk, subject, condition, seed, device)
