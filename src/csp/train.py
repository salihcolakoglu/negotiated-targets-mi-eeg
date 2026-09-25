#!/usr/bin/env python3
"""Experiment 1: five equal-budget target rules on the paper's CSP inputs.

Scientific implementation: Claude (Anthropic, Cowork), deney01.py, 2026-09-24,
using the original project's model and Walsh routines (Nuri Korhan/coauthors).
Portable CLI packaging: GPT-6 Astra / Codex, 2026-09-25.

C0 hard targets; C1 cumulative negotiation; C2 label-anchored negotiation;
C3 label-anchored negotiation gated by correct training predictions;
C4 smoothing towards 0.5. K1 uses Walsh rows 0..K-1; K2 uses rows 1..K (D1 only).
All conditions use dropout 0.1, 10 phases x 20 epochs, SGD and batch size 16.
The original fit/predict/update/predict call order is retained. Test data passed
as validation_data are monitored only: no early stopping or model selection.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np

try:
    from .config import DATASETS, BATCH_SIZE, data_path, selected_subjects
    from .utils import set_seeds, build_cnn_model_original, create_walsh
except ImportError:
    from config import DATASETS, BATCH_SIZE, data_path, selected_subjects
    from utils import set_seeds, build_cnn_model_original, create_walsh

R_RATE, PHASES, EPOCHS, DROPOUT = 0.03, 10, 20, 0.1
W = create_walsh(16)
CONDS = ['C0', 'C1', 'C2', 'C3', 'C4']
SEEDS = [1, 2, 3]


def code_rows(n_cls, code):
    rows = list(range(n_cls)) if code == 'K1' else list(range(1, n_cls + 1))
    return W[rows].astype(np.float32)


def nearest(P, C):
    d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=-1)
    return d.argmin(1)


def bce(P, Y):
    P = np.clip(P.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean())


def load_csp(work_dir, dk, s):
    with data_path(work_dir, dk, s).open('rb') as fh:
        return pickle.load(fh)


def run_one(work_dir, dk, di, s, cond, code, seed):
    set_seeds(seed)
    x_tr, y_tr, x_te, y_te = load_csp(work_dir, dk, s)
    x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
    x_te = x_te.reshape(x_te.shape[0], x_te.shape[1], 1)
    n = di['n_classes']
    C = code_rows(n, code)
    ytr0, yte0 = np.asarray(y_tr) - 1, np.asarray(y_te) - 1
    Y = C[ytr0].copy()
    Yte = C[yte0].copy()
    model = build_cnn_model_original(x_tr.shape[1:], dropout_rate=DROPOUT)
    Yneg = Y.copy()
    Y_used = Yneg
    phase_acc = []
    t0 = time.time()
    for p in range(PHASES):
        Y_used = Yneg
        model.fit(x_tr, Yneg, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(x_te, Yte))
        P = model.predict(x_tr, verbose=0)
        a = R_RATE * p
        if cond == 'C1':
            Yneg = Yneg - (Yneg - P) * R_RATE * p
        elif cond == 'C2':
            Yneg = (1 - a) * Y + a * P
        elif cond == 'C3':
            ok = nearest(P, C) == ytr0
            Yneg = Y.copy()
            Yneg[ok] = (1 - a) * Y[ok] + a * P[ok]
        elif cond == 'C4':
            Yneg = ((1 - a) * Y + a * 0.5).astype(np.float32)
        Pte = model.predict(x_te, verbose=0)
        phase_acc.append(float((nearest(Pte, C) == yte0).mean()))
    Pte = model.predict(x_te, verbose=0)
    Ptr = model.predict(x_tr, verbose=0)
    rec = {
        'dataset': dk, 'subject': s, 'condition': cond, 'code': code, 'seed': seed,
        'acc_test': float((nearest(Pte, C) == yte0).mean()),
        'acc_train': float((nearest(Ptr, C) == ytr0).mean()),
        'bce_test': bce(Pte, Yte),
        'last_target_wrong_frac': float((nearest(Y_used, C) != ytr0).mean()),
        'last_target_dist': float(np.linalg.norm(Y_used - Y, axis=1).mean()),
        'phase_test_acc': phase_acc,
        'seconds': round(time.time() - t0, 1),
    }
    return rec


def key(d):
    return f"{d['dataset']}|{d['subject']}|{d['condition']}|{d['code']}|{d['seed']}"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_identity(work_dir, plan):
    """Bind resumable outputs to their exact inputs, scientific code and environment."""
    inputs = {(dk, s) for dk, s, *_ in plan}
    source_dir = Path(__file__).resolve().parent
    versions = {}
    for name in ('tensorflow', 'keras', 'numpy', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'schema': 1,
        'inputs_sha256': {f'{dk}/S{s:02d}': file_sha256(data_path(work_dir, dk, s))
                          for dk, s in sorted(inputs)},
        'code_sha256': {name: file_sha256(source_dir / name)
                        for name in ('train.py', 'config.py', 'utils.py', 'features.py', 'prepare.py')},
        'versions': versions,
        'selected_plan': [list(t) for t in plan],
    }


def reject_nonfinite_json(value):
    raise ValueError(f'Non-finite JSON value is not permitted: {value}')


def make_plan(datasets=None, subjects=None, conditions=None, codes=None, seeds=None):
    """Filter the original 615-run order without reordering selected runs."""
    plan = []
    for dk, di in DATASETS.items():
        for s in range(1, di['n_subjects'] + 1):
            plan.append((dk, s, 'C1', 'K1', 1))
    for seed in SEEDS:
        for cond in CONDS:
            for dk, di in DATASETS.items():
                available = ['K1', 'K2'] if di['n_classes'] == 4 else ['K1']
                for code in available:
                    for s in range(1, di['n_subjects'] + 1):
                        t = (dk, s, cond, code, seed)
                        if t not in plan:
                            plan.append(t)
    return [t for t in plan
            if (datasets is None or t[0] in datasets)
            and (subjects is None or t[1] in subjects)
            and (conditions is None or t[2] in conditions)
            and (codes is None or t[3] in codes)
            and (seeds is None or t[4] in seeds)]


def compare_reference(done, reference, strict=False):
    """Report differences, permitting platform variation unless strict is requested."""
    rows = []
    for rec in done.values():
        if (rec['condition'], rec['code'], rec['seed']) != ('C1', 'K1', 1):
            continue
        expected = reference[rec['dataset']]['per_subject'][rec['subject'] - 1]
        delta = rec['acc_test'] - expected
        rows.append({'dataset': rec['dataset'], 'subject': rec['subject'],
                     'observed': rec['acc_test'], 'reference': expected, 'difference': delta,
                     'within_1e_9': abs(delta) <= 1e-9})
    bad = [r for r in rows if not r['within_1e_9']]
    print(f'Reference comparison: {len(rows)} available C1/K1/seed-1 runs; {len(bad)} differ beyond 1e-9', flush=True)
    for row in bad:
        print(f"  {row['dataset']} S{row['subject']:02d}: observed={row['observed']:.9f}, "
              f"reference={row['reference']:.9f}, delta={row['difference']:+.9f}", flush=True)
    if strict and bad:
        raise RuntimeError('Strict reference check failed; saved runs have been retained.')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--max-runs', type=int, default=0, help='Maximum new runs (0 = all selected runs)')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--conditions', nargs='+', choices=CONDS, default=CONDS)
    parser.add_argument('--codes', nargs='+', choices=['K1', 'K2'], default=['K1', 'K2'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--reference', type=Path, help='Optional REPRO T08_negotiated_results.json, not the original published-run result')
    parser.add_argument('--strict-reference', action='store_true', help='Abort if available C1/K1/seed-1 accuracies differ by more than 1e-9')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without TensorFlow, file writes, downloads or training')
    args = parser.parse_args(argv)
    if args.max_runs < 0:
        parser.error('--max-runs must be non-negative')
    if args.strict_reference and args.reference is None:
        parser.error('--strict-reference requires --reference')
    for dk in args.datasets:
        selected_subjects(dk, args.subjects)
    plan = make_plan(args.datasets, args.subjects, args.conditions, args.codes, args.seeds)
    if not plan:
        parser.error('No runs selected (K2 applies to D1 only).')
    if args.dry_run:
        print(json.dumps({'n_runs': len(plan), 'first_runs': plan[:10],
                          'dropout': DROPOUT, 'phases': PHASES, 'epochs_per_phase': EPOCHS}, indent=2))
        return
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    for dk, s, *_ in plan:
        if not data_path(args.work_dir, dk, s).exists():
            parser.error(f'Missing CSP input: {data_path(args.work_dir, dk, s)}')
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / 'deney01_kosular.jsonl'
    identity_path = args.out / 'run_identity.json'
    identity = run_identity(args.work_dir, plan)
    if identity_path.exists():
        previous_identity = json.loads(identity_path.read_text(encoding='utf-8'))
        if previous_identity != identity:
            parser.error('Output identity differs (inputs, code, environment or selected plan). Use a new --out directory; existing results are preserved.')
    else:
        if jsonl.exists():
            parser.error('Existing JSONL has no run_identity.json. Preserve it and use a new --out directory.')
        with identity_path.open('x', encoding='utf-8') as fh:
            json.dump(identity, fh, indent=2, allow_nan=False)
            fh.write('\n')
    done = {}
    if jsonl.exists():
        for line_number, line in enumerate(jsonl.read_text(encoding='utf-8').splitlines(), 1):
            if line.strip():
                try:
                    record = json.loads(line, parse_constant=reject_nonfinite_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid existing JSONL line {line_number}; preserve it and use a fresh output directory.') from exc
                record_key = key(record)
                if record_key in done:
                    raise ValueError(f'Duplicate run key at JSONL line {line_number}: {record_key}. Existing results are preserved.')
                done[record_key] = record
    reference = json.loads(args.reference.read_text(encoding='utf-8')) if args.reference else None
    if reference is not None:
        compare_reference(done, reference, args.strict_reference)
    new = 0
    print(f'Plan: {len(plan)} runs; already completed: {sum("|".join(map(str, t)) in done for t in plan)}', flush=True)
    for t in plan:
        k = '|'.join(map(str, t))
        if k in done:
            continue
        dk, s, cond, code, seed = t
        rec = run_one(args.work_dir, dk, DATASETS[dk], s, cond, code, seed)
        with jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        done[k] = rec
        new += 1
        print(f"RUN {k} acc={rec['acc_test']:.4f} train={rec['acc_train']:.4f} {rec['seconds']}s", flush=True)
        if reference is not None and (cond, code, seed) == ('C1', 'K1', 1):
            rows = compare_reference(done, reference, False)
            (args.out / 'reference_comparison.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
            if args.strict_reference and any(not row['within_1e_9'] for row in rows):
                raise RuntimeError('Strict reference check failed; saved runs have been retained.')
        if args.max_runs and new >= args.max_runs:
            print('Maximum new-run count reached; restart the same command to resume.', flush=True)
            return
    print('All selected Experiment 1 runs are complete.', flush=True)


if __name__ == '__main__':
    main()
