#!/usr/bin/env python3
"""Download/cache the study datasets, preserve its splits, and construct CSP inputs.

Adapted from the original project's T04/T05 and the corrected REPRO_v1.0 T04
mapping (Claude, 2026-09-24). Packaging: GPT-6 Astra / Codex, 2026-09-25.
This deliberately retains the paper's one-ended CSP feature selection.
Only load pickle files that you generated or obtained from a trusted source.
"""
import argparse
import csv
import json
import os
import pickle
from pathlib import Path
import numpy as np

try:
    from .config import DATASETS, RANDOM_SEED, data_path, selected_subjects
    from .features import bandpass_filter, compute_csp
except ImportError:
    from config import DATASETS, RANDOM_SEED, data_path, selected_subjects
    from features import bandpass_filter, compute_csp


def split_subject(X, labels, meta, subject):
    """Original chronological split and one-indexed sorted class mapping.

    D1/D4: first and last session; middle sessions are not used.
    D3: the first and last half of its single session (no random split).
    D4 subject 8 has 160 training trials; most D4 subjects have 120.
    """
    subj_mask = meta['subject'] == subject
    sessions = meta.loc[subj_mask, 'session'].values
    unique_sessions = sorted(set(sessions))
    if len(unique_sessions) >= 2:
        train_mask = subj_mask & (meta['session'] == unique_sessions[0])
        test_mask = subj_mask & (meta['session'] == unique_sessions[-1])
    else:
        subj_indices = np.where(subj_mask)[0]
        mid = len(subj_indices) // 2
        train_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[subj_indices[:mid]] = True
        test_mask[subj_indices[mid:]] = True
    x_train = X[train_mask]
    x_test = X[test_mask]
    y_train_raw = labels.values[train_mask] if hasattr(labels, 'values') else labels[train_mask]
    y_test_raw = labels.values[test_mask] if hasattr(labels, 'values') else labels[test_mask]
    all_classes = sorted(set(y_train_raw) | set(y_test_raw))
    class_map = {c: i + 1 for i, c in enumerate(all_classes)}
    y_train = np.array([class_map[c] for c in y_train_raw])
    y_test = np.array([class_map[c] for c in y_test_raw])
    return (x_train, y_train, x_test, y_test), class_map, unique_sessions


def prepare_raw(work_dir, data_dir, datasets, subjects=None, overwrite=False):
    # Environment-scoped cache settings avoid machine-specific hardcoded paths.
    cache = str(Path(data_dir).resolve())
    os.environ['MNE_DATA'] = cache
    os.environ['MNE_DATASETS_BNCI_PATH'] = cache
    Path(cache).mkdir(parents=True, exist_ok=True)
    import mne
    from moabb.datasets import BNCI2014_001, BNCI2014_002, BNCI2014_004
    from moabb.paradigms import MotorImagery, LeftRightImagery
    mne.set_log_level('WARNING')
    # These are the corrected, validated T04 mappings, not the archived swap.
    definitions = {
        'D1': (BNCI2014_001, lambda: MotorImagery(n_classes=4)),
        'D3': (BNCI2014_002, lambda: MotorImagery(n_classes=2)),
        'D4': (BNCI2014_004, LeftRightImagery),
    }
    summary = []
    for dk in datasets:
        requested = selected_subjects(dk, subjects)
        pending = [s for s in requested if overwrite or not data_path(work_dir, dk, s, raw=True).exists()]
        if not pending:
            print(f'{dk}: reuse {len(requested)} existing raw-array files')
            continue
        dataset_type, paradigm_type = definitions[dk]
        # The default study path intentionally makes the same get_data call as T04.
        X, labels, meta = paradigm_type().get_data(dataset=dataset_type())
        found_subjects = sorted(meta['subject'].unique())
        found_classes = sorted(labels.unique()) if hasattr(labels, 'unique') else sorted(np.unique(labels))
        if len(found_subjects) != DATASETS[dk]['n_subjects'] or len(found_classes) != DATASETS[dk]['n_classes']:
            raise ValueError(f'{dk}: unexpected subjects/classes: {len(found_subjects)}/{len(found_classes)}')
        for subject in pending:
            arrays, mapping, sessions = split_subject(X, labels, meta, subject)
            path = data_path(work_dir, dk, subject, raw=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as fh:
                pickle.dump(arrays, fh)
            x_train, y_train, x_test, y_test = arrays
            row = {'dataset': dk, 'subject': subject, 'train_shape': list(x_train.shape),
                   'test_shape': list(x_test.shape), 'sessions': [str(x) for x in sessions],
                   'class_map': {str(k): v for k, v in mapping.items()}}
            summary.append(row)
            print(f'{dk} S{subject:02d}: raw train={x_train.shape}, test={x_test.shape}')
    if summary:
        out = Path(work_dir) / 'preparation'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'raw_created.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def prepare_csp(work_dir, datasets, subjects=None, overwrite=False):
    np.random.seed(RANDOM_SEED)
    rows = []
    for dk in datasets:
        di = DATASETS[dk]
        for subject in selected_subjects(dk, subjects):
            source = data_path(work_dir, dk, subject, raw=True)
            target = data_path(work_dir, dk, subject)
            reused = target.exists() and not overwrite
            if reused:
                with target.open('rb') as fh:
                    x_train_csp, y_train, x_test_csp, y_test = pickle.load(fh)
            else:
                if not source.exists():
                    raise FileNotFoundError(f'Run the raw preparation stage first: {source}')
                with source.open('rb') as fh:
                    x_train, y_train, x_test, y_test = pickle.load(fh)
                x_train_filt = bandpass_filter(x_train, low=8.0, high=30.0, fs=float(di['fs']))
                x_test_filt = bandpass_filter(x_test, low=8.0, high=30.0, fs=float(di['fs']))
                n_comp = 6 if di['n_classes'] == 2 else 4 * di['n_classes']
                try:
                    x_train_csp, x_test_csp = compute_csp(x_train_filt, y_train, x_test_filt, n_components=n_comp)
                except Exception as e:
                    # Preserve the original numerical fallback; expose it in the log.
                    print(f'WARNING {dk} S{subject:02d}: CSP error {e}; using variance features')
                    x_train_csp = np.log(np.var(x_train_filt, axis=2) + 1e-10)
                    x_test_csp = np.log(np.var(x_test_filt, axis=2) + 1e-10)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as fh:
                    pickle.dump((x_train_csp, y_train, x_test_csp, y_test), fh)
            row = {'dataset': dk, 'subject': subject, 'train_samples': len(y_train),
                   'test_samples': len(y_test), 'n_features': x_train_csp.shape[1],
                   'n_classes': len(np.unique(y_train)), 'csp_file': target.name, 'reused': reused}
            rows.append(row)
            print(f'{dk} S{subject:02d}: CSP train={x_train_csp.shape}, test={x_test_csp.shape}'
                  + (' (reused)' if reused else ''))
    out = Path(work_dir) / 'preparation'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'csp_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path, help='MOABB/MNE download cache')
    parser.add_argument('--stage', choices=['all', 'raw', 'csp'], default='all')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--subjects', nargs='+', type=int, help='Optional subject IDs, valid in every selected dataset')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild selected generated arrays; otherwise reuse existing files')
    args = parser.parse_args(argv)
    for dk in args.datasets:
        selected_subjects(dk, args.subjects)
    if args.stage in ('all', 'raw'):
        prepare_raw(args.work_dir, args.data_dir, args.datasets, args.subjects, args.overwrite)
    if args.stage in ('all', 'csp'):
        prepare_csp(args.work_dir, args.datasets, args.subjects, args.overwrite)


if __name__ == '__main__':
    main()
