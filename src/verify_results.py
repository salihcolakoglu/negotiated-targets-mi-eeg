#!/usr/bin/env python3
"""Verify the stored CSP-CNN and EEGNet results using Python and NumPy.

No model training, project imports, network requests, pickle loading, or input writes.
Numerical beta/t/Holm routines adapted from the independent GPT/Codex v4 audit
script, not from the original experiment/training code. Input paths are relative
to this script. Run: python verify_supplement_v4.py [--report OUTSIDE_PACKAGE.json]
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import platform
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
NS = {'D1': 9, 'D3': 14, 'D4': 9}
PARTICIPANTS = [(d, s) for d, n in NS.items() for s in range(1, n + 1)]
PAIRS = {'E1': [('A1', 'C1', 'C0'), ('A2', 'C3', 'C0'),
                ('A3', 'C3', 'C1'), ('A4', 'C2', 'C4')],
         'E2': [('A1', 'W1', 'W0'), ('A2', 'W2', 'W0'),
                ('A3', 'W2', 'W1'), ('A4', 'W2', 'W3')]}
METRICS = ['acc_test', 'acc_train', 'bce_test', 'nll_test', 'ece_test',
           'last_target_dist', 'last_target_wrong_frac']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(name):
    return json.loads((BASE / name).read_text(encoding='utf-8-sig'))


def load_csv(name):
    with (BASE / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-10, label='numeric comparison'):
    error = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    require(math.isfinite(error) and error <= tolerance, f'{label}: difference {error}')
    return error


def beta_cf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            return h
    raise ArithmeticError('Incomplete beta failed to converge')


def ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    q = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return q * beta_cf(a, b, x) / a
    return 1 - q * beta_cf(b, a, 1 - x) / b


def t_p(t, df):
    return ibeta(df / 2, 0.5, df / (df + t * t))


def tcrit95(df):
    lo, hi = 0.0, 128.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if t_p(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holm(values):
    order = np.argsort(values)
    result, last = [0.0] * len(values), 0.0
    for rank, index in enumerate(order):
        last = min(1.0, max(last, values[index] * (len(values) - rank)))
        result[int(index)] = last
    return result


def paired(a, b):
    diff = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * 100
    n, mean, sd = len(diff), float(diff.mean()), float(diff.std(ddof=1))
    require(n > 1 and sd > 0 and np.isfinite(diff).all(), 'Invalid paired-test input')
    se = sd / math.sqrt(n)
    stat = mean / se
    margin = tcrit95(n - 1) * se
    return {'n': n, 'mean_diff_pp': mean, 'ci95_pp': [mean - margin, mean + margin],
            'p_t': t_p(stat, n - 1), 't': stat, 'd_z': mean / sd,
            'sd_diff_pp': sd, 'higher': int(np.sum(diff > 1e-12)),
            'ties': int(np.sum(abs(diff) <= 1e-12)), 'lower': int(np.sum(diff < -1e-12))}


def verify_manifest():
    listed = set()
    for line in (BASE / 'MANIFEST_SHA256.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (BASE / relative).resolve()
        require(path.is_relative_to(BASE) and path.is_file(), 'Invalid manifest path')
        require(relative not in listed, 'Duplicate manifest entry')
        require(sha(path) == expected, f'Hash mismatch: {relative}')
        listed.add(relative)
    actual = {p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()
              and p.name != 'MANIFEST_SHA256.txt' and '__pycache__' not in p.parts}
    require(actual == listed, 'Manifest and package contents differ')
    return len(listed)


def expected_run_keys():
    e1 = {('E1', d, s, c, k, seed) for d, n in NS.items() for s in range(1, n + 1)
          for c in ['C0', 'C1', 'C2', 'C3', 'C4']
          for k in (['K1', 'K2'] if d == 'D1' else ['K1']) for seed in [1, 2, 3]}
    e2 = {('E2', d, s, c, 'softmax' if c == 'S0' else 'K1', seed)
          for d, n in NS.items() for s in range(1, n + 1)
          for c in ['S0', 'W0', 'W1', 'W2', 'W3'] for seed in [1, 2, 3, 4, 5]}
    return e1 | e2


def run_key(row):
    return (row['experiment'], row['dataset'], int(row['subject']), row['condition'],
            row['code'], int(row['seed']))


def verify_tables(summary):
    rows = load_csv('e1_e2_all_1415_runs_verified_v4.csv')
    keys = [run_key(r) for r in rows]
    require(len(keys) == len(set(keys)) == 1415 and set(keys) == expected_run_keys(),
            'Run CSV is incomplete or duplicated')
    runs = dict(zip(keys, rows))
    groups = collections.defaultdict(list)
    for key, row in runs.items():
        groups[key[:-1]].append(row)
    subject_rows = load_csv('e1_e2_subject_seed_means_verified_v4.csv')
    require(len(subject_rows) == len(groups) == 365, 'Subject CSV count mismatch')
    values, seen = {}, set()
    for row in subject_rows:
        key = (row['experiment'], row['dataset'], int(row['subject']), row['condition'], row['code'])
        require(key in groups and key not in seen, 'Invalid subject CSV key')
        seen.add(key)
        rr = groups[key]
        test = [float(r['acc_test']) for r in rr]
        train = [float(r['acc_train']) for r in rr]
        require(int(row['n_seeds']) == len(rr), 'Seed count mismatch')
        close(np.mean(test), row['mean_test_accuracy'], label='Subject test mean')
        close(np.mean(train), row['mean_train_accuracy'], label='Subject train mean')
        close(np.std(test, ddof=1) * 100, row['test_seed_sd_pp'], label='Subject seed SD')
        values[key] = float(np.mean(test))

    def score(exp, condition, dataset, subject):
        return values[(exp, dataset, subject, condition, 'softmax' if condition == 'S0' else 'K1')]

    pair_rows = load_csv('e1_e2_paired_participant_differences_verified_v4.csv')
    expected_pairs = {(exp, label, d, s) for exp, pairs in PAIRS.items()
                      for label, a, b in pairs for d, s in PARTICIPANTS}
    expected_pairs |= {('E2', 'descriptive', d, s) for d, s in PARTICIPANTS}
    actual_pairs = set()
    for row in pair_rows:
        exp, label, d, s = row['experiment'], row['label'], row['dataset'], int(row['subject'])
        key = exp, label, d, s
        require(key not in actual_pairs, 'Duplicate paired CSV row')
        actual_pairs.add(key)
        known = {lab: (a, b) for lab, a, b in PAIRS[exp]}
        if exp == 'E2':
            known['descriptive'] = ('W0', 'S0')
        require((row['condition_a'], row['condition_b']) == known[label], 'Contrast mismatch')
        a, b = [score(exp, row[f'condition_{c}'], d, s) for c in ['a', 'b']]
        close(a, row['mean_accuracy_a']); close(b, row['mean_accuracy_b'])
        close((a - b) * 100, row['difference_pp'])
    require(actual_pairs == expected_pairs and len(pair_rows) == 288, 'Paired CSV mismatch')
    all_results, maximum = {}, 0.0
    for exp, pairs in PAIRS.items():
        results = []
        for label, a, b in pairs:
            result = paired([score(exp, a, d, s) for d, s in PARTICIPANTS],
                            [score(exp, b, d, s) for d, s in PARTICIPANTS])
            result.update(label=label, contrast=f'{a} - {b}')
            results.append(result)
        for result, adjusted, expected in zip(results, holm([r['p_t'] for r in results]),
                                              summary[exp + '_main_family']):
            result['p_holm'] = adjusted
            require(result['contrast'] == expected['contrast'], 'Summary contrast mismatch')
            for field in ['mean_diff_pp', 'ci95_pp', 'p_t', 'p_holm', 'd_z', 't', 'sd_diff_pp']:
                maximum = max(maximum, close(result[field], expected[field], 1e-8, exp + ' ' + field))
            for field in ['n', 'higher', 'ties', 'lower']:
                require(result[field] == expected[field], 'Summary count mismatch')
        all_results[exp] = results
    return runs, all_results, maximum


def compare_sanitized(runs, name, exp):
    records = load_json(name)
    seen = set()
    for row in records:
        key = run_key(row)
        require(row['experiment'] == exp and key in runs and key not in seen, 'Invalid clean record')
        seen.add(key)
        for field in METRICS:
            expected = runs[key][field]
            if expected == '':
                require(row.get(field) is None, 'Expected missing metric')
            else:
                close(row[field], expected, label='Clean record ' + field)
    require(len(seen) == (615 if exp == 'E1' else 800), 'Sanitized record count mismatch')
    return records


def ece(probability, labels):
    confidence = probability.max(1)
    correct = probability.argmax(1) == labels
    total, bounds = 0.0, np.linspace(0, 1, 16)
    for i in range(15):
        mask = (confidence >= bounds[i] if i == 0 else confidence > bounds[i]) & (confidence <= bounds[i + 1])
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def verify_predictions(records):
    walsh = np.ones((1, 1), dtype=float)
    for _ in range(4):
        walsh = np.block([[walsh, walsh], [walsh, 1 - walsh]])
    classify = lambda p, c: ((p[:, None, :] - c[None, :, :]) ** 2).sum(2).argmin(1)
    errors = collections.defaultdict(float)
    label_cache, files = {}, set()

    def metric(name, actual, expected):
        errors[name] = max(errors[name], close(actual, expected, 2e-7, name))

    for row in records:
        name = row['npz_file']
        require(Path(name).name == name and name.endswith('.npz'), 'NPZ name must be a basename')
        require(name not in files, 'Duplicate NPZ file reference')
        files.add(name)
        path = BASE / 'eegnet_predictions' / name
        require(sha(path) == row['npz_sha256'], 'NPZ hash mismatch')
        c = walsh[:4 if row['dataset'] == 'D1' else 2]
        with np.load(path, allow_pickle=False) as z:
            for field in z.files:
                require(z[field].dtype.kind != 'O', 'Object array rejected')
            for subset in ['train', 'test']:
                p, y = z['P_' + subset].astype(float), z['y_' + subset]
                require(p.ndim == 2 and y.ndim == 1 and len(p) == len(y) == row['n_' + subset], 'NPZ shape mismatch')
                require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
                require(y.dtype.kind in 'iu' and ((y >= 0) & (y < len(c))).all(), 'Invalid labels')
                key = row['dataset'], row['subject'], subset
                if key in label_cache:
                    require(np.array_equal(y, label_cache[key]), 'Participant label mismatch')
                else:
                    label_cache[key] = y.copy()
                pred = p.argmax(1) if row['condition'] == 'S0' else classify(p, c)
                metric('acc_' + subset, float(np.mean(pred == y)), row['acc_' + subset])
                if row['condition'] == 'S0':
                    post = p
                else:
                    q = np.clip(p, 1e-7, 1 - 1e-7)
                    logl = np.sum(c[None, :, :] * np.log(q[:, None, :])
                                  + (1 - c[None, :, :]) * np.log1p(-q[:, None, :]), axis=2)
                    post = np.exp(logl - logl.max(1, keepdims=True))
                    post /= post.sum(1, keepdims=True)
                    if subset == 'test':
                        metric('bce_test', float(-np.mean(c[y] * np.log(q) + (1 - c[y]) * np.log1p(-q))), row['bce_test'])
                nll = float(-np.log(np.clip(post[np.arange(len(y)), y], 1e-12, 1)).mean())
                metric('nll_' + subset, nll, row['nll_' + subset])
                if subset == 'test':
                    metric('ece_test', ece(post, y), row['ece_test'])
                    metric('acc_test_post', float(np.mean(post.argmax(1) == y)), row['acc_test_post'])
            if row['condition'] != 'S0':
                y, target = z['y_train'], z['Y_used_last'].astype(float)
                metric('last_target_wrong_frac', float(np.mean(classify(target, c) != y)), row['last_target_wrong_frac'])
                metric('last_target_dist', float(np.linalg.norm(target - c[y], axis=1).mean()), row['last_target_dist'])
    require(len(files) == 800, 'Expected 800 prediction files')
    require(files == {p.name for p in (BASE / 'eegnet_predictions').glob('*.npz')}, 'Extra/missing prediction files')
    return dict(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional NEW report path outside this package')
    parser.add_argument('--quiet', action='store_true', help='Save the report without printing its full JSON')
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted reference-data directory")
    args = parser.parse_args()
    global BASE
    BASE = args.data_dir.resolve()
    close(tcrit95(31), 2.0395134463964078, 1e-10, 't critical-value sanity check')
    close(t_p(0.0, 31), 1.0)
    close(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    manifest_count = verify_manifest()
    summary = load_json('verified_e1_e2_tables_v4.json')
    runs, families, max_error = verify_tables(summary)
    compare_sanitized(runs, 'e1_records_sanitized_v4.json', 'E1')
    e2 = compare_sanitized(runs, 'e2_records_sanitized_v4.json', 'E2')
    raw_errors = verify_predictions(e2)
    result = {'status': 'PASS', 'python': platform.python_version(), 'numpy': np.__version__,
              'manifest_files_verified': manifest_count, 'training_runs': 1415,
              'independent_participants': 32, 'subject_condition_code_rows': 365,
              'paired_participant_rows': 288, 'main_paired_comparisons_verified': 8,
              'prediction_files_hash_verified': 800, 'prediction_files_metrics_verified': 800,
              'max_main_statistic_absolute_error': max_error,
              'max_prediction_metric_absolute_errors': raw_errors, 'main_families': families,
              'scope': 'Data/analysis consistency only; no model training, raw EEG preprocessing audit, or equivalence claim.'}
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.report:
        target = args.report.resolve()
        require(not target.is_relative_to(BASE), 'Report must be outside the package')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded)
    if not args.quiet:
        print(encoded, end='')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
