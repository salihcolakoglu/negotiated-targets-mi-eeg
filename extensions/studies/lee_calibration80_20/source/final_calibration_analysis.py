"""Complete-case paired exploratory calibration analysis; never partial results."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from calibration_runner import verify_protocol, verify_global_manifest, job_key, jobs, sha, atomic_json, canonical_hash
from calibration_math import fit_temperature, legacy_class_scores, log_posterior
from lee_data import code_rows

METRICS = ['nll_unclipped', 'nll_legacy_clipped', 'ece_15_equal_width', 'brier_sum_over_classes',
           'accuracy_original_decoder', 'accuracy_class_posterior', 'decoder_posterior_disagreement_fraction']


def independent_metrics(log_scores, y, temperature, decoded):
    # Direct recomputation from retained scores/labels; JSON metrics are not data.
    z = np.asarray(log_scores, dtype=np.float64) / temperature
    lp = z - logsumexp(z, axis=1, keepdims=True)
    prob = np.exp(lp)
    pred = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)
    correct = pred == y
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        selected = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - conf[selected].mean())
    return dict(nll_unclipped=float(-lp[np.arange(len(y)), y].mean()),
                nll_legacy_clipped=float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean()),
                ece_15_equal_width=float(ece),
                brier_sum_over_classes=float(np.square(prob - np.eye(2)[y]).sum(axis=1).mean()),
                accuracy_original_decoder=float(np.mean(decoded == y)),
                accuracy_class_posterior=float(np.mean(pred == y)),
                decoder_posterior_disagreement_fraction=float(np.mean(decoded != pred)))


def paired_contrast(d):
    d = np.asarray(d, dtype=np.float64)
    assert d.shape == (54,) and np.isfinite(d).all()
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(.975, 53) * sd / np.sqrt(54))
    if sd == 0:
        p = 1.0 if mean == 0 else 0.0
    else:
        p = float(stats.ttest_1samp(d, 0).pvalue)
    return {'delta_calibrated_minus_raw': mean, 'pointwise_95ci': [mean-half, mean+half],
            'p_raw': p, 'n_participants': 54, 'ci_is_multiplicity_adjusted': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--protocol-sha256', required=True)
    args = ap.parse_args()
    p = verify_protocol(args.protocol, args.protocol_sha256)
    for rel, digest in p['analysis_code_sha256'].items():
        if sha(ROOT/rel) != digest:
            raise RuntimeError('Analysis source changed: '+rel)
    frozen, frozen_sha = verify_global_manifest(p)
    completion = json.loads((ROOT/'test_results/COMPLETION.json').read_text(encoding='utf-8'))
    assert completion['count'] == completion['expected'] == 1350
    assert completion['global_manifest_sha256'] == frozen_sha
    expected = {job_key(*j)+'.json' for j in jobs()}
    assert {q.name for q in (ROOT/'test_results/records').glob('*.json')} == expected
    entries = {e['job']:e for e in frozen['entries']}
    rows = []
    temperature_rows = []
    for subject, condition, seed in jobs():
        key = job_key(subject, condition, seed)
        rpath = ROOT/'test_results/records'/(key+'.json')
        r = json.loads(rpath.read_text(encoding='utf-8'))
        assert r['status'] == 'PAIRED_RAW_CALIBRATED_TEST_COMPLETE' and r['n_test'] == 100
        assert r['job'] == key and r['prediction_file'] == 'predictions/'+key+'.npz'
        assert r['identity']['global_manifest_sha256'] == frozen_sha
        assert r['identity']['fit_identity']['subject'] == subject
        assert r['identity']['fit_identity']['condition'] == condition
        assert r['identity']['fit_identity']['seed'] == seed
        assert r['identity']['fit_identity']['protocol_sha256'] == args.protocol_sha256
        fpath = ROOT/'fits/records'/(key+'.json')
        fr = json.loads(fpath.read_text(encoding='utf-8'))
        assert sha(fpath) == r['identity']['fit_record_sha256']
        assert fr['identity'] == r['identity']['fit_identity']
        with np.load(ROOT/'fits/calibration'/(key+'.npz'), allow_pickle=False) as cal:
            fit_idx, cal_idx = cal['fit_indices'], cal['calibration_indices']
            assert len(fit_idx) == 80 and len(cal_idx) == 20 and not np.intersect1d(fit_idx, cal_idx).size
            assert np.bincount(cal['y_calibration'],minlength=2).tolist() == [10,10]
            cal_scores = legacy_class_scores(cal['P_calibration'],walsh=condition != 'S0',codes=code_rows())
            assert np.allclose(cal_scores,cal['class_log_scores_calibration'],rtol=0,atol=1e-12)
            fitted = fit_temperature(cal['class_log_scores_calibration'], cal['y_calibration'])
        temp = float(r['temperature'])
        assert np.isclose(temp, fitted['temperature'], atol=1e-10, rtol=1e-10)
        predfile = ROOT/'test_results'/r['prediction_file']
        assert sha(predfile) == r['prediction_sha256']
        with np.load(predfile, allow_pickle=False) as a:
            y = a['y_test']
            assert y.dtype == np.int64 and y.shape == (100,) and np.bincount(y).tolist() == [50,50]
            P = a['P_network']
            decoded = a['original_decoder_predictions']
            C = code_rows()
            expected_decoded = ((P[:, None, :]-C[None, :, :])**2).sum(axis=-1).argmin(axis=1) if condition != 'S0' else P.argmax(axis=1)
            assert np.array_equal(decoded, expected_decoded)
            scores = legacy_class_scores(P, walsh=condition != 'S0', codes=C)
            assert np.allclose(scores, a['class_log_scores_test'], rtol=0, atol=1e-12)
            assert float(a['temperature']) == temp
            assert np.allclose(np.exp(log_posterior(scores)), a['posterior_raw'], rtol=0, atol=1e-12)
            assert np.allclose(np.exp(log_posterior(scores,temp)), a['posterior_calibrated'], rtol=0, atol=1e-12)
            assert np.array_equal(a['posterior_raw'].argmax(1),a['posterior_calibrated'].argmax(1))
            for mode,T in [('raw',1.), ('calibrated',temp)]:
                metrics = independent_metrics(scores, y, T, decoded)
                for m,v in metrics.items():
                    assert np.isclose(v,r[mode][m],rtol=0,atol=1e-10), (key,mode,m)
                rows.append(dict(subject=subject, condition=condition, seed=seed, mode=mode, **metrics))
        assert r['raw']['accuracy_original_decoder'] == r['calibrated']['accuracy_original_decoder']
        temperature_rows.append(dict(subject=subject, condition=condition, seed=seed, temperature=temp,
                                     boundary=fitted['boundary'], chosen_candidate=fitted['chosen_candidate']))
    participant = []
    summary = []
    for condition in p['scientific_config']['conditions']:
        for subject in range(1,55):
            for mode in ['raw','calibrated']:
                group = [r for r in rows if r['condition']==condition and r['subject']==subject and r['mode']==mode]
                assert len(group)==5 and {r['seed'] for r in group}==set(range(1,6))
                participant.append(dict(subject=subject,condition=condition,mode=mode,**{m:float(np.mean([r[m] for r in group])) for m in METRICS}))
        cs = dict(condition=condition, metrics={})
        for m in METRICS:
            raw = np.array([r[m] for r in participant if r['condition']==condition and r['mode']=='raw'])
            calibrated = np.array([r[m] for r in participant if r['condition']==condition and r['mode']=='calibrated'])
            stat = paired_contrast(calibrated-raw)
            if m != 'nll_unclipped':
                stat.pop('p_raw')  # Secondary outcomes remain descriptive.
            cs['metrics'][m] = dict(raw_mean=float(raw.mean()),calibrated_mean=float(calibrated.mean()),**stat)
        temps=[t for t in temperature_rows if t['condition']==condition]
        cs['temperatures']=dict(min=min(t['temperature'] for t in temps),median=float(np.median([t['temperature'] for t in temps])),max=max(t['temperature'] for t in temps),boundary_count=sum(t['boundary'] for t in temps),models=len(temps))
        summary.append(cs)
    order = np.argsort([s['metrics']['nll_unclipped']['p_raw'] for s in summary])
    running=0.
    for rank,i in enumerate(order):
        rawp=summary[i]['metrics']['nll_unclipped']['p_raw']
        running=max(running,min(1.,(5-rank)*rawp))
        summary[i]['metrics']['nll_unclipped']['p_holm_5_calibration_contrasts']=running
    out=ROOT/'final_analysis'
    out.mkdir(exist_ok=True)
    for filename,data in [('run_metrics.csv',rows),('participant_metrics.csv',participant),('temperatures.csv',temperature_rows)]:
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    results=dict(status='PASS_COMPLETE_EXPLORATORY_CALIBRATION_ANALYSIS',protocol_sha256=args.protocol_sha256,
        global_fit_manifest_sha256=frozen_sha, fits=1350, participants=54, independent_unit='mean of five seeds per participant',
        primary='within-condition calibrated-minus-raw stable NLL; Holm family of exactly five conditions',
        secondary='ECE/Brier and clipped NLL descriptive; original decoder accuracy unchanged',
        old_100_trial_results_not_a_paired_control=True, earlier_lee_results_already_seen=True,
        no_new_independent_confirmatory_claim=True, summary=summary, completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'CALIBRATION_RESULTS-V1.0.json',results)
    text=['KALİBRASYON KONTROLÜ — V1.0','Durum: bütün 1.350 yeni model doğrulandı; keşif amaçlı değerlendirme.','Her kişi beş tohum ortalamasıyla tek gözlem; n=54.','Kalibrasyon eğitimden ayrılmış 20 kayıtta öğrenildi; aynı yeni 80 kayıtla eğitilmiş modelin önce/sonra çifti karşılaştırıldı.','Pozitif sıcaklık sınıf kararını değiştirmez; doğruluk artışı iddia edilmez.','Negatif NLL/ECE/Brier farkı iyileşme demektir. Noktasal %95 aralıklar çoklu karşılaştırma için düzeltilmiş aralık değildir.','']
    for s in summary:
        n=s['metrics']['nll_unclipped'];e=s['metrics']['ece_15_equal_width'];b=s['metrics']['brier_sum_over_classes']
        text.append(f"{s['condition']}: NLL {n['raw_mean']:.4f} → {n['calibrated_mean']:.4f}; fark {n['delta_calibrated_minus_raw']:+.4f}; Holm p={n['p_holm_5_calibration_contrasts']:.6g}. ECE {e['raw_mean']:.4f} → {e['calibrated_mean']:.4f}; Brier {b['raw_mean']:.4f} → {b['calibrated_mean']:.4f}. Sınırdaki sıcaklık:{s['temperatures']['boundary_count']}/270.")
    text+=['','Bu kontrol olumluysa yalnız eğitim içi ayarlanan güven dönüşümünün yeni gün olasılık ölçülerine aktarımını destekler; bit bağımlılığının sebep olduğunu tek başına kanıtlamaz. Olumsuz veya karışık sonuçlar da korunur. Mevcut IEEE Access V1.2 makalesi ve eski istatistik aileleri değişmedi.']
    (out/'KALIBRASYON_SONUC_OZETI-V1.0.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    atomic_json(ROOT/'STATUS.json',dict(status='COMPLETE_ANALYSIS_VERIFIED',completed=1350,expected=1350,protocol_sha256=args.protocol_sha256,result=str(out/'CALIBRATION_RESULTS-V1.0.json'),updated_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status=results['status'],fits=1350,participants=54),ensure_ascii=False))


if __name__=='__main__':
    main()
