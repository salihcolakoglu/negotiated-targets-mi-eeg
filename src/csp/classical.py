#!/usr/bin/env python3
"""CSP+LDA reference with the exact scikit-learn defaults used in the manuscript.

Extracted from analyses by Claude (Anthropic, Cowork); portable packaging by
GPT-6 Astra / Codex, 2026-09-25. Fit on training CSP features and score later
recordings. No parameter search or additional feature scaling is performed.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

try:
    from .config import DATASETS, data_path, selected_subjects
except ImportError:
    from config import DATASETS, data_path, selected_subjects


def run(work_dir, datasets=None, subjects=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    records = []
    for dk in DATASETS if datasets is None else datasets:
        for subject in selected_subjects(dk, subjects):
            with data_path(work_dir, dk, subject).open('rb') as fh:
                xtr, ytr, xte, yte = pickle.load(fh)
            lda = LinearDiscriminantAnalysis().fit(xtr, ytr)
            row = {'dataset': dk, 'subject': subject,
                   'lda_train': float((lda.predict(xtr) == ytr).mean()),
                   'lda_test': float((lda.predict(xte) == yte).mean()),
                   'n_train': len(ytr), 'n_test': len(yte), 'n_features': xtr.shape[1]}
            records.append(row)
            print(f"{dk} S{subject:02d}: LDA accuracy={row['lda_test']:.6f}")
    means = {dk: float(np.mean([r['lda_test'] for r in records if r['dataset'] == dk]))
             for dk in DATASETS if any(r['dataset'] == dk for r in records)}
    return {'per_subject': records, 'dataset_means': means}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--subjects', nargs='+', type=int)
    args = parser.parse_args(argv)
    for dk in args.datasets:
        selected_subjects(dk, args.subjects)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'classical_results.json'
    if path.exists():
        parser.error(f'Output already exists; use a fresh --out directory: {path}')
    result = run(args.work_dir, args.datasets, args.subjects)
    with path.open('x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
        fh.write('\n')
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
