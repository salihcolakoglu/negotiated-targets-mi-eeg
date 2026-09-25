"""EEGNet, target updates and metrics used in the 800-run experiment.
Scientific function bodies are retained from the archived implementation.
Walsh accuracy uses nearest codewords; NLL/ECE use restricted Bernoulli posteriors.
"""
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import R_RATE, PHASES

def _renorm_(w, maxnorm):
    with torch.no_grad():
        w.data = torch.renorm(w.data, p=2, dim=0, maxnorm=maxnorm)


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018) for 128 Hz input; max-norm 1 on the depthwise spatial filters and 0.25 on
    the classifier, as in the reference implementation."""

    def __init__(self, n_ch, n_times, n_out, dropout=0.5, F1=8, D=2, F2=16, kern=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.dw = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            n_feat = self.features(torch.zeros(2, n_ch, n_times)).shape[1]
        self.fc = nn.Linear(n_feat, n_out)

    def features(self, x):
        x = self.bn1(self.conv1(x.unsqueeze(1)))
        x = F.elu(self.bn2(self.dw(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.sep_pw(self.sep_dw(x))))
        x = self.drop2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))

    def constrain(self):
        _renorm_(self.dw.weight, 1.0)
        _renorm_(self.fc.weight, 0.25)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch(threads=2):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nearest(P, C):
    d = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def ece_score(prob, y0, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    ok = (pred == y0).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def walsh_metrics(P, y0, C):
    """P: sigmoid outputs (n, 16). Accuracy by nearest codeword (paper), bit BCE against the true codeword,
    and a class posterior from independent Bernoulli bits restricted to the K codewords (for NLL and ECE)."""
    P = P.astype(np.float64)
    pred = nearest(P, C)
    Pc = np.clip(P, 1e-7, 1 - 1e-7)
    Y = C[y0]
    bce = float(-(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)).mean())
    logp = np.log(Pc) @ C.T.astype(np.float64) + np.log(1 - Pc) @ (1 - C.T.astype(np.float64))
    logp -= logp.max(1, keepdims=True)
    post = np.exp(logp)
    post /= post.sum(1, keepdims=True)
    n = len(y0)
    return dict(acc=float((pred == y0).mean()), bce=bce,
                nll=float(-np.log(np.clip(post[np.arange(n), y0], 1e-12, 1.0)).mean()),
                ece=ece_score(post, y0), acc_post=float((post.argmax(1) == y0).mean()),
                pred=[int(v) for v in pred])


def softmax_metrics(P, y0):
    P = P.astype(np.float64)
    pred = P.argmax(1)
    n = len(y0)
    return dict(acc=float((pred == y0).mean()), bce=None,
                nll=float(-np.log(np.clip(P[np.arange(n), y0], 1e-12, 1.0)).mean()),
                ece=ece_score(P, y0), acc_post=float((pred == y0).mean()),
                pred=[int(v) for v in pred])


def _batches(n, batch, gen):
    perm = torch.randperm(n, generator=gen)
    chunks = list(perm.split(batch))
    if len(chunks) > 1 and len(chunks[-1]) < 2:          # BatchNorm needs >= 2 samples per batch
        chunks[-2] = torch.cat([chunks[-2], chunks[-1]])
        chunks.pop()
    return chunks


def train_epochs(model, opt, X_t, T_t, n_epochs, batch, gen, walsh):
    model.train()
    n = X_t.shape[0]
    dev = X_t.device
    for _ in range(n_epochs):
        for idx in _batches(n, batch, gen):
            idx = idx.to(dev)
            logits = model(X_t[idx])
            if walsh:
                loss = F.binary_cross_entropy_with_logits(logits, T_t[idx])
            else:
                loss = F.cross_entropy(logits, T_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.constrain()


@torch.no_grad()
def predict(model, X_t, walsh, bs=256):
    model.eval()
    out = []
    for i in range(0, X_t.shape[0], bs):
        out.append(model(X_t[i:i + bs]))
    z = torch.cat(out).float()
    return (torch.sigmoid(z) if walsh else torch.softmax(z, 1)).cpu().numpy()


def fit_phases(model, Xtr, ytr0, Xte, yte0, *, walsh, rule, C, lr, epochs_per_phase, batch, seed, dev):
    """Phase loop with the same call order as REPRO T08 / Deney 01: fit on the current target, predict the
    training trials, update the target, evaluate the test trials."""
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if walsh:
        Y = C[ytr0].astype(np.float32)
        Yneg = Y.copy()
    else:
        Y = Yneg = None
    Y_used = Yneg
    Cbar = C.mean(0, keepdims=True).astype(np.float32) if walsh else None
    gate = None
    phase_acc, used_dist, used_gate = [], [], []
    t0 = time.time()
    for p in range(PHASES):
        Y_used = Yneg
        if walsh:
            used_dist.append(float(np.linalg.norm(Y_used - Y, axis=1).mean()))
            used_gate.append(None if gate is None else float(gate.mean()))
        T_t = torch.from_numpy(Yneg).to(dev) if walsh else torch.from_numpy(ytr0.astype(np.int64)).to(dev)
        train_epochs(model, opt, Xtr_t, T_t, epochs_per_phase, batch, gen, walsh)
        P = predict(model, Xtr_t, walsh)
        a = R_RATE * p
        if walsh and rule == "neg_paper":
            Yneg = (Yneg - (Yneg - P) * R_RATE * p).astype(np.float32)
        elif walsh and rule == "neg_anchor_gate":
            gate = nearest(P, C) == ytr0
            Yneg = Y.copy()
            Yneg[gate] = ((1 - a) * Y[gate] + a * P[gate]).astype(np.float32)
        elif walsh and rule == "smooth_class":
            Yneg = ((1 - a) * Y + a * Cbar).astype(np.float32)
        elif rule != "hard":
            raise ValueError(rule)
        Pte = predict(model, Xte_t, walsh)
        phase_acc.append(float((nearest(Pte, C) == yte0).mean()) if walsh else float((Pte.argmax(1) == yte0).mean()))
    Pte = predict(model, Xte_t, walsh)
    Ptr = predict(model, Xtr_t, walsh)
    mte = walsh_metrics(Pte, yte0, C) if walsh else softmax_metrics(Pte, yte0)
    mtr = walsh_metrics(Ptr, ytr0, C) if walsh else softmax_metrics(Ptr, ytr0)
    rec = dict(acc_test=mte["acc"], acc_train=mtr["acc"], acc_test_post=mte["acc_post"],
               bce_test=mte["bce"], nll_test=mte["nll"], ece_test=mte["ece"], nll_train=mtr["nll"],
               phase_test_acc=phase_acc, pred_test=mte["pred"], train_seconds=round(time.time() - t0, 2))
    arrays = dict(P_test=Pte.astype(np.float32), P_train=Ptr.astype(np.float32), y_test=yte0.astype(np.int64),
                  y_train=ytr0.astype(np.int64))
    if walsh:
        rec["last_target_wrong_frac"] = float((nearest(Y_used, C) != ytr0).mean())
        rec["last_target_dist"] = float(np.linalg.norm(Y_used - Y, axis=1).mean())
        rec["used_target_dist_by_phase"] = used_dist
        rec["used_gate_open_by_phase"] = used_gate
        arrays["Y_used_last"] = Y_used.astype(np.float32)
    return rec, arrays



def build_model(name, n_ch, n_times, n_out, dropout):
    if name != "EEGNet":
        raise ValueError("This package contains only the completed EEGNet experiment")
    return EEGNet(n_ch, n_times, n_out, dropout=dropout)
