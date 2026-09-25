#!/usr/bin/env python3
"""Download/extract EEGNet input and verify the original E1 trial-label split.

Run --help without NumPy/MNE/MOABB installed. Data are downloaded from their
public providers through MOABB, not bundled with this repository.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-raw', type=Path, required=True,
                        help='E1 Raw_Datasets folder produced by the data-preparation step')
    parser.add_argument('--data-dir', type=Path, required=True, help='New output folder')
    parser.add_argument('--cache-dir', type=Path, required=True, help='MOABB/MNE download cache')
    return parser.parse_args()


def main():
    args = arguments()
    if args.data_dir.exists() and any(args.data_dir.iterdir()):
        raise SystemExit('Output must be absent or empty; existing data are never overwritten. '
                         'After an interrupted preparation, choose a new --data-dir; '
                         'the download cache can be reused without downloading completed source files again.')
    prefixes = [('D1_BCIIV2a', 9), ('D3_BNCI2014_002', 14), ('D4_BCIIV2b', 9)]
    missing = [str(args.reference_raw / f'{prefix}_S{s:02d}.pkl')
               for prefix, count in prefixes for s in range(1, count + 1)
               if not (args.reference_raw / f'{prefix}_S{s:02d}.pkl').is_file()]
    if missing:
        raise SystemExit(f'First prepare the 32 E1 raw trial files. Missing: {missing[:3]}')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MNE_DATA'] = str(args.cache_dir.resolve())
    import numpy as np
    import mne
    import moabb
    from moabb.datasets import BNCI2014_001, BNCI2014_002, BNCI2014_004
    from moabb.paradigms import MotorImagery, LeftRightImagery
    # Environment overrides prevent an older machine-specific MNE configuration
    # from selecting a different cache. Numerical preprocessing is unchanged.
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(args.cache_dir.resolve())
    mne.set_log_level('WARNING')
    specs = [('D1', 'D1_BCIIV2a', BNCI2014_001(), MotorImagery(n_classes=4, fmin=4, fmax=38, resample=128)),
             ('D3', 'D3_BNCI2014_002', BNCI2014_002(), MotorImagery(n_classes=2, fmin=4, fmax=38, resample=128)),
             ('D4', 'D4_BCIIV2b', BNCI2014_004(), LeftRightImagery(fmin=4, fmax=38, resample=128))]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    summary, written = {}, []
    limit = 19 * 1024 * 1024  # Retain the archived partitioning rule.
    for dk, prefix, dataset, paradigm in specs:
        X, labels, meta = paradigm.get_data(dataset=dataset)
        labels = np.asarray(labels)
        classes = sorted(set(labels))
        class_map = {c: i + 1 for i, c in enumerate(classes)}
        for subject in sorted(meta['subject'].unique()):
            mask = (meta['subject'] == subject).values
            sessions = meta.loc[mask, 'session'].values
            runs = meta.loc[mask, 'run'].values
            xs = X[mask].astype(np.float32)
            ys = np.array([class_map[c] for c in labels[mask]])
            unique_sessions = sorted(set(sessions))
            if len(unique_sessions) >= 2:
                train = sessions == unique_sessions[0]
                test = sessions == unique_sessions[-1]
            else:
                idx = np.arange(len(ys))
                train = idx < len(ys) // 2
                test = idx >= len(ys) // 2
            # These trusted pickles are generated locally by this repository.
            # Never substitute a downloaded/untrusted pickle file.
            reference = args.reference_raw / f'{prefix}_S{subject:02d}.pkl'
            with reference.open('rb') as stream:
                _, y_train_ref, _, y_test_ref = pickle.load(stream)
            if not (np.array_equal(ys[train], y_train_ref) and np.array_equal(ys[test], y_test_ref)):
                raise ValueError(f'{prefix} S{subject:02d}: trial-label sequence differs from E1')
            # Same conversion and array insertion order as veri_donustur.py.
            xs = np.ascontiguousarray(xs, dtype=np.float32)
            n_parts = max(1, math.ceil(xs.nbytes / limit))
            bounds = np.linspace(0, xs.shape[0], n_parts + 1).round().astype(int)
            metadata = dict(y=np.asarray(ys, dtype=np.int64), train_mask=np.asarray(train, dtype=bool),
                            test_mask=np.asarray(test, dtype=bool),
                            session=np.asarray([str(v) for v in sessions]),
                            run=np.asarray([str(v) for v in runs]))
            base = f'{prefix}_S{subject:02d}'
            for k in range(n_parts):
                a, b = int(bounds[k]), int(bounds[k + 1])
                path = args.data_dir / f'{base}_p{k + 1}.npz'
                np.savez(path, X=xs[a:b], part=np.int64(k + 1), n_parts=np.int64(n_parts),
                         trial_start=np.int64(a), **metadata)
                written.append(path)
            summary[base] = dict(shape=list(xs.shape), n_parts=n_parts,
                                 train=int(train.sum()), test=int(test.sum()),
                                 sessions=sorted(set(metadata['session'].tolist())),
                                 matches_paper_split=True, x_sha256=hashlib.sha256(xs.tobytes()).hexdigest())
            print(f'{dk} S{subject:02d}: {xs.shape}, train/test={train.sum()}/{test.sum()}, split verified', flush=True)
    if len(summary) != 32:
        raise ValueError(f'Expected 32 participants, got {len(summary)}')
    # A completed manifest is written only after all 32 participants succeeded.
    (args.data_dir / 'VERI_OZET.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    hashes = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in sorted(written)]
    (args.data_dir / 'VERI_SHA256.txt').write_text('\n'.join(hashes) + '\n', encoding='utf-8')
    receipt = {'numpy': np.__version__, 'mne': mne.__version__, 'moabb': moabb.__version__,
               'participants': len(summary), 'npz_files': len(written), 'band_hz': [4, 38], 'sampling_hz': 128,
               'split_check': 'Exact training/test label sequence equality against locally prepared E1 trials'}
    (args.data_dir / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(summary)} participants. No training was performed.')


if __name__ == '__main__':
    main()
