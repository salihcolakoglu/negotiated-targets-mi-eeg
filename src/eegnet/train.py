#!/usr/bin/env python3
"""Run the 800 EEGNet jobs; completed jobs are verified and resumed.

CPU is the default, matching the recorded study execution. --max-jobs limits
the number of new full-length runs. --smoke uses one epoch per phase solely
to exercise the pipeline, and must use a separate output folder.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
DEVICE = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize(threads):
    global DEVICE
    from core import setup_torch
    DEVICE = setup_torch(threads)


def work(item):
    job, data_dir = item
    from experiment import run_job
    start = time.time()
    try:
        outputs = run_job(job, data_dir, DEVICE)
        seconds = round(time.time() - start, 2)
        for record, _ in outputs:
            record['seconds'] = seconds
            record['device'] = str(DEVICE)
        return job, outputs, None
    except Exception:
        return job, None, traceback.format_exc()


def strict(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('A non-finite output was produced; no completed record was written.')
    if isinstance(value, dict):
        return {k: strict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict(v) for v in value]
    return value


def read_done(path, identity):
    from experiment import rec_key
    done = set()
    if not path.exists():
        return done
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        record = json.loads(line)
        key = rec_key(record)
        if key in done or record.get('identity') != identity:
            raise ValueError(f'Cannot resume: duplicate key or changed experiment identity on line {index}. Use a new --out.')
        prediction = path.parent / record['cikti_npz']
        if not prediction.is_file() or sha(prediction) != record['cikti_sha256']:
            raise ValueError(f'Cannot resume: missing or changed prediction file for {key}')
        done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=0, help='Maximum new jobs; 0 means all 800')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--smoke', action='store_true', help='One epoch/phase; not manuscript results')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.max_jobs < 0:
        parser.error('workers/threads must be positive and max-jobs non-negative')
    os.environ['NEGREP_SMOKE'] = '1' if args.smoke else '0'
    os.environ.setdefault('PYTHONHASHSEED', '0')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from experiment import job_list, rec_key
    from data import DATASETS
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is not available; use --device cpu')
    data_dir = args.data_dir.resolve()
    manifest = data_dir / 'VERI_SHA256.txt'
    lines = manifest.read_text(encoding='utf-8').splitlines()
    checked, listed = 0, set()
    for line in lines:
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip('*')
        if name in listed:
            raise ValueError(f'Duplicate data manifest entry: {name}')
        listed.add(name)
        path = (data_dir / name).resolve()
        if not path.is_relative_to(data_dir) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Data integrity check failed: {name}')
        checked += 1
    summary = json.loads((data_dir / 'VERI_OZET.json').read_text(encoding='utf-8'))
    expected_subjects = {f'{di["prefix"]}_S{s:02d}' for di in DATASETS.values()
                         for s in range(1, di['n_subjects'] + 1)}
    if set(p.name for p in data_dir.glob('*.npz')) != listed:
        raise ValueError('Data directory contains missing or unlisted NPZ files; use a clean prepared directory')
    if checked < 32 or set(summary) != expected_subjects or not all(v['matches_paper_split'] for v in summary.values()):
        raise ValueError('Expected prepared data for 32 participants with verified splits')
    identity = {'code_sha256': {name: sha(HERE / name) for name in ['train.py', 'core.py', 'data.py', 'experiment.py']},
                'data_list_sha256': sha(manifest), 'smoke': args.smoke, 'device': args.device,
                'threads': args.threads, 'numpy': np.__version__, 'torch': torch.__version__}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'kosular.jsonl'
    done = read_done(path, identity)
    jobs = job_list()
    pending = [job for job in jobs if '|'.join(map(str, job)) not in done]
    if args.max_jobs:
        pending = pending[:args.max_jobs]
    if args.smoke and not args.max_jobs:
        pending = pending[:1]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    receipt = {'identity': identity, 'python': platform.python_version(), 'platform': platform.platform(),
               'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
               'workers': args.workers, 'jobs_requested': len(pending)}
    receipt_path = args.out / f'environment_{stamp}_{time.time_ns()}.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(f'{len(done)}/800 complete; {len(pending)} new jobs; {args.device}; smoke={args.smoke}', flush=True)
    if not pending:
        return
    failures = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=initialize, initargs=(args.threads,)) as pool:
        for index, (job, outputs, error) in enumerate(pool.imap_unordered(work, [(j, str(data_dir)) for j in pending], chunksize=1), 1):
            if error:
                failures.append('|'.join(map(str, job)))
                print(error, flush=True)
                continue
            for record, arrays in outputs:
                key = rec_key(record)
                dest = args.out / 'cikti' / (key.replace('|', '__') + '.npz')
                dest.parent.mkdir(exist_ok=True)
                temporary = dest.with_name(dest.name + '.tmp.npz')
                np.savez(temporary, **arrays)
                os.replace(temporary, dest)
                record.update(identity=identity, cikti_npz='cikti/' + dest.name, cikti_sha256=sha(dest))
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(strict(record), allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f'{index}/{len(pending)} {key}: accuracy={record["acc_test"]:.4f}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} jobs failed; rerun the same command to resume successful output safely.')
    print(f'Complete records: {len(read_done(path, identity))}/800', flush=True)


if __name__ == '__main__':
    main()
