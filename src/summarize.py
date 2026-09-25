#!/usr/bin/env python3
"""Summarize a complete new 615-run CSP and 800-run EEGNet experiment."""
from pathlib import Path
import argparse
import collections
import csv
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1', type=Path, required=True)
    parser.add_argument('--e2', type=Path, required=True)
    parser.add_argument('--classical', type=Path, help='Optional complete 32-person CSP-LDA results')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    import numpy as np
    from verify_results import expected_run_keys, paired, holm, PAIRS, PARTICIPANTS, NS
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new output directory to preserve previous summaries.')
    records = {}
    e2_identities = set()
    e2_has_identity = []
    for experiment, path in [('E1', args.e1), ('E2', args.e2)]:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if experiment == 'E2':
                if row.get('experiment') not in ['D02','E2'] or row.get('model') != 'EEGNet':
                    raise ValueError('The E2 file must contain EEGNet Experiment 2 records only.')
                e2_has_identity.append('identity' in row)
                if 'identity' in row:
                    e2_identities.add(json.dumps(row['identity'],sort_keys=True))
            if row.get('identity', {}).get('smoke') or row.get('smoke'):
                raise ValueError('Smoke-test records cannot be summarized as study results.')
            code = row.get('code', 'softmax' if row['condition'] == 'S0' else 'K1')
            key = (experiment, row['dataset'], int(row['subject']), row['condition'], code, int(row['seed']))
            if key in records:
                raise ValueError(f'Duplicate run: {key}')
            if not all(np.isfinite(float(row[k])) and 0 <= float(row[k]) <= 1 for k in ['acc_train','acc_test']):
                raise ValueError(f'Invalid accuracy: {key}')
            records[key] = row
    expected = expected_run_keys()
    if len(e2_identities)>1 or (any(e2_has_identity) and not all(e2_has_identity)):
        raise ValueError('Mixed EEGNet run identities; do not combine different training configurations.')
    if set(records) != expected:
        missing, extra = expected - set(records), set(records) - expected
        raise ValueError(f'Expected all 1415 study runs; missing={len(missing)}, extra={len(extra)}. '
                         'Complete the full runs before producing manuscript-level summaries.')
    groups = collections.defaultdict(list)
    for key, row in records.items():
        groups[key[:-1]].append(row)
    means = {key:float(np.mean([r['acc_test'] for r in rows])) for key,rows in groups.items()}
    def scores(exp, cond):
        code = 'softmax' if cond == 'S0' else 'K1'
        return [means[(exp,d,s,cond,code)] for d,s in PARTICIPANTS]
    families = {}
    for exp,pairs in PAIRS.items():
        results = []
        for label,a,b in pairs:
            value = paired(scores(exp,a),scores(exp,b))
            value.update(label=label,contrast=f'{a} - {b}')
            results.append(value)
        for value,p in zip(results,holm([v['p_t'] for v in results])):
            value['p_holm'] = p
        families[exp] = results
    sensitivity = []
    for cond in ['C0','C1','C2','C3','C4']:
        a = [means[('E1','D1',s,cond,'K2')] for s in range(1,10)]
        b = [means[('E1','D1',s,cond,'K1')] for s in range(1,10)]
        value = paired(a,b);value['condition'] = cond;sensitivity.append(value)
    classical = []
    if args.classical:
        ref = json.loads(args.classical.read_text(encoding='utf-8'))['per_subject']
        lookup = {(r['dataset'],int(r['subject'])):float(r['lda_test']) for r in ref}
        if len(ref) != 32 or set(lookup) != set(PARTICIPANTS) or not all(np.isfinite(x) and 0<=x<=1 for x in lookup.values()):
            raise ValueError('Expected one valid LDA result for every participant.')
        for d,n in NS.items():
            for cond in ['S0','W2']:
                code = 'softmax' if cond=='S0' else 'K1'
                values = [lookup[(d,s)] for s in range(1,n+1)]
                result = paired([means[('E2',d,s,cond,code)] for s in range(1,n+1)],values)
                result.update(dataset=d,condition=cond,lda_mean=float(np.mean(values)))
                classical.append(result)
        for value,p in zip(classical,holm([v['p_t'] for v in classical])):value['p_holm']=p
    calibration = []
    for d in NS:
        for cond in ['S0','W0','W1','W2','W3']:
            values = [r for k,r in records.items() if k[0]=='E2' and k[1]==d and k[3]==cond]
            row = dict(dataset=d,condition=cond)
            for name in ['nll_test','ece_test','last_target_dist','last_target_wrong_frac']:
                vals = [float(r[name]) for r in values if r.get(name) is not None]
                if not all(np.isfinite(x) for x in vals):raise ValueError(f'Nonfinite {name}')
                row[name] = float(np.mean(vals)) if vals else None
            calibration.append(row)
    output = {'runs':len(records),'participants':32,'families':families,
              'K2_minus_K1':sensitivity,'classical_reference':classical,'EEGNet_probability_measures':calibration,
              'EEGNet_W0_minus_S0_exploratory':paired(scores('E2','W0'),scores('E2','S0'))}
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'participant_means.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['experiment','dataset','subject','condition','code','n_seeds','accuracy'])
        for key in sorted(means):w.writerow([*key,len(groups[key]),means[key]])
    (args.out/'paired_comparisons.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    text = ['# Recomputed study tables','',
            'Accuracy is a percentage; contrasts are percentage points. Participants, not seeds, are the analysis unit.','']
    for exp,conditions in [('E1',['C0','C1','C2','C3','C4']),('E2',['S0','W0','W1','W2','W3'])]:
        text += [f'## {exp} accuracy','', '| Condition | D1 | D3 | D4 |','|---|---:|---:|---:|']
        for cond in conditions:
            code = 'softmax' if cond == 'S0' else 'K1'
            values = [np.mean([means[(exp,d,s,cond,code)] for s in range(1,NS[d]+1)])*100 for d in NS]
            text.append('| '+cond+' | '+' | '.join(f'{x:.2f}' for x in values)+' |')
        text += ['',f'## {exp} paired comparisons','',
                 '| Contrast | Difference (pp) | Pointwise 95% CI | Holm p |',
                 '|---|---:|---|---:|']
        for value in families[exp]:
            ci=value['ci95_pp']
            text.append(f"| {value['contrast']} | {value['mean_diff_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {value['p_holm']:.3f} |")
        text.append('')
    text += ['## Code selection (D1)', '', '| Condition | K2−K1 (pp) | Pointwise 95% CI | Raw p |', '|---|---:|---|---:|']
    for r in sensitivity:
        lo,hi=r['ci95_pp'];text.append(f"| {r['condition']} | {r['mean_diff_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['p_t']:.3f} |")
    text += ['', '## EEGNet probability measures', '', '| Dataset | Condition | NLL | ECE |', '|---|---|---:|---:|']
    for r in calibration:text.append(f"| {r['dataset']} | {r['condition']} | {r['nll_test']:.3f} | {r['ece_test']:.3f} |")
    if classical:
        text += ['', '## Classical reference', '', '| Dataset | Condition | LDA (%) | Difference (pp) | Pointwise 95% CI | Holm p |', '|---|---|---:|---:|---|---:|']
        for r in classical:
            lo,hi=r['ci95_pp'];text.append(f"| {r['dataset']} | {r['condition']} | {100*r['lda_mean']:.2f} | {r['mean_diff_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['p_holm']:.3f} |")
    text += ['', 'Four contrasts are corrected separately within each experiment; the six classical-reference comparisons form a separate family. Code-selection p values are unadjusted. Confidence intervals are not multiplicity-adjusted.', '']
    (args.out/'tables.md').write_text('\n'.join(text),encoding='utf-8')
    print(f'Wrote participant means, paired comparisons and tables to {args.out.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)
