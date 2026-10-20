"""Complete-case independent analysis of the four-method SHU comparison.

No EEG values, network checkpoints or training APIs are deserialized. All
identities/hashes must pass before any observed prediction metric is calculated.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import joblib
import numpy as np
from scipy.special import expit,logsumexp
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[21,6,10,18,19,25,8,7,15,12]
EVALUATION=[22,14,17,4,1,16,2,5,11,20,9,24,3,23,13]
METHOD_CONFIG={'A':('EEGNet','CE',256),'B':('EEGNet','WALSH',256),
               'C':('FBCNet','CE',1152),'D':('FBCNet','WALSH',1152)}
CLASSICAL=('SINGLE_CSP_LDA','BANK_CSP_LDA')
METHODS=tuple(METHOD_CONFIG)+CLASSICAL
SEEDS=(1,2,3,4,5);DAYS=(2,3,4,5)
COUNTS={'cv_neural_fits':80,'main_neural_fits':320,'classical_fits':50,
        'target_temperature_records':330,'cv_validation_records':60,'tests':1320}
METRICS=('accuracy','balanced_accuracy','recall_left','recall_right','nll','ece15','brier',
         'posterior_accuracy','decoder_posterior_disagreement_fraction','mean_max_probability')

def require(condition,message):
    if not bool(condition):raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def local(root,relative):
    require(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'Expected relative local artifact')
    p=(Path(root)/relative).resolve()
    require(p.is_relative_to(Path(root).resolve()),'Artifact escapes study: '+relative)
    return p

def moment(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(t.tzinfo is not None,'Timestamp lacks timezone')
    return t

def array_sha(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(json.dumps(list(a.shape),separators=(',',':')).encode('ascii'))
    h.update(a.tobytes());return h.hexdigest()

def labels(value,n=None,balanced=None):
    y=np.asarray(value)
    require(y.dtype==np.int64 and y.ndim==1 and (n is None or len(y)==n) and set(y.tolist())=={0,1},'Invalid balanced/binary label array')
    if balanced is not None:require(np.bincount(y,minlength=2).tolist()==[balanced,balanced],'Wrong class counts')
    return y

def checked_scores(value):
    x=np.asarray(value)
    require(x.dtype==np.float64 and x.ndim==2 and x.shape[1]==2 and len(x)>0 and np.isfinite(x).all(),'Expected finite float64 Nx2 class scores')
    return x

def native_scores(logits,probabilities,output):
    l=np.asarray(logits);q=np.asarray(probabilities)
    width=2 if output=='CE' else 16
    require(output in ('CE','WALSH') and l.dtype==q.dtype==np.float32 and l.ndim==2 and l.shape[1]==width and q.shape==l.shape and np.isfinite(l).all() and np.isfinite(q).all(),'Wrong retained native output shape/dtype')
    require(np.all((q>=0)&(q<=1)),'Native probabilities outside unit interval')
    if output=='CE':
        scores=l.astype(np.float64)
        expected=np.exp(scores-logsumexp(scores,axis=1,keepdims=True))
        require(np.allclose(expected,q,rtol=0,atol=2e-7),'CE softmax/logits mismatch')
        prediction=l.argmax(1).astype(np.int64)
    else:
        p=expit(l.astype(np.float64));require(np.allclose(p,q,rtol=0,atol=1e-7),'Walsh sigmoid/logits mismatch')
        p=np.clip(p,1e-7,1-1e-7)
        code=np.asarray([[1.]*16,[1.,0.]*8],np.float64)
        scores=np.log(p)@code.T+np.log1p(-p)@(1-code.T)
        prediction=np.square(q[:,None,:]-code.astype(np.float32)[None,:,:]).sum(2).argmin(1).astype(np.int64)
    return scores,prediction

def probability_metrics(scores,y,original_pred,T=1.):
    scores=checked_scores(scores);y=labels(y,len(scores));pred=np.asarray(original_pred)
    require(pred.dtype==np.int64 and pred.shape==y.shape and np.isin(pred,[0,1]).all(),'Wrong retained decision array')
    require(np.isfinite(T) and T>0,'Invalid positive temperature')
    scaled=scores/float(T);require(np.isfinite(scaled).all(),'Scaled score overflow')
    lp=scaled-logsumexp(scaled,axis=1,keepdims=True);p=np.exp(lp)
    posterior=scores.argmax(1);confidence=p.max(1);correct=posterior==y
    ece=0.;edges=np.linspace(0,1,16)
    for k in range(15):
        keep=((confidence>=edges[k]) if k==0 else (confidence>edges[k]))&(confidence<=edges[k+1])
        if keep.any():ece+=keep.mean()*abs(correct[keep].mean()-confidence[keep].mean())
    return dict(accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean([(pred[y==c]==c).mean() for c in (0,1)])),
        recall_left=float((pred[y==0]==0).mean()),recall_right=float((pred[y==1]==1).mean()),
        nll=float(-lp[np.arange(len(y)),y].mean()),ece15=float(ece),brier=float(np.square(p-np.eye(2)[y]).sum(1).mean()),
        posterior_accuracy=float((posterior==y).mean()),decoder_posterior_disagreement_fraction=float((pred!=posterior).mean()),
        mean_max_probability=float(confidence.mean()),n=int(len(y)),n_class0=int((y==0).sum()),n_class1=int((y==1).sum()),
        n_correct=int((pred==y).sum()),n_posterior_correct=int((posterior==y).sum()),n_disagreement=int((pred!=posterior).sum()))

def close_metrics(stored,actual,label):
    require(set(stored)==set(actual),'Metric schema mismatch '+label)
    for name,value in actual.items():
        require(np.isfinite(stored[name]) and np.isclose(stored[name],value,rtol=1e-10,atol=1e-11),'Metric mismatch '+label+'/'+name)

def paired_difference_pp(value):
    d=np.asarray(value,np.float64)
    require(d.shape==(15,) and np.isfinite(d).all(),'Contrasts require exactly15 participant means')
    mean,sd=float(d.mean()),float(d.std(ddof=1));half=float(stats.t.ppf(.975,14)*sd/np.sqrt(15))
    p=(1. if mean==0 else 0.) if sd==0 else float(stats.ttest_1samp(d,0.).pvalue)
    return dict(n_participants=15,df=14,delta_mean_pp=mean,pointwise_95ci_pp=[mean-half,mean+half],
        ci_is_multiplicity_adjusted=False,p_two_sided=p,participant_differences_pp=d.tolist(),
        participants_improved=int((d>0).sum()),participants_deteriorated=int((d<0).sum()),participants_tied=int((d==0).sum()))

def holm5(values):
    p=np.asarray(values,np.float64)
    require(p.shape==(5,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Exactlyfive finite p-values required')
    order=np.argsort(p,kind='stable');a=np.minimum(1.,np.maximum.accumulate((5-np.arange(5))*p[order]));out=np.empty(5);out[order]=a
    return out

def summarize(rows):
    lookup={(r['subject'],r['method'],r['seed'],r['day'],r['variant']):r for r in rows}
    require(len(lookup)==len(rows),'Duplicate metric identity')
    require(len(rows)==3240,'Expected3240 raw/calibrated/T8 metric rows')
    people=[];summary=[]
    for day in DAYS:
        for method in METHODS:
            seeds=SEEDS if method in METHOD_CONFIG else (0,)
            variants=('raw','calibrated','T8') if method in ('B','D') else ('raw','calibrated')
            for variant in variants:
                pp=[]
                for subject in EVALUATION:
                    rr=[lookup[(subject,method,k,day,variant)] for k in seeds]
                    require(all((r['n'],r['n_class0'],r['n_class1'])==(rr[0]['n'],rr[0]['n_class0'],rr[0]['n_class1']) for r in rr),'Trial identities differ across starts')
                    values={name:float(np.mean([r[name] for r in rr])) for name in METRICS}
                    for metric,count in [('accuracy','n_correct'),('posterior_accuracy','n_posterior_correct'),('decoder_posterior_disagreement_fraction','n_disagreement')]:
                        values[metric]=sum(r[count] for r in rr)/(len(seeds)*rr[0]['n'])
                    row=dict(subject=subject,method=method,day=day,variant=variant,seeds=len(seeds),
                        n_test_each_seed=rr[0]['n'],n_class0=rr[0]['n_class0'],n_class1=rr[0]['n_class1'],
                        temperature_mean=float(np.mean([r['temperature'] for r in rr])),**values)
                    pp.append(row);people.append(row)
                summary.append(dict(method=method,day=day,variant=variant,n_participants=15,seeds_per_person=len(seeds),
                    **{name:float(np.mean([r[name] for r in pp])) for name in METRICS},
                    **{name+'_sd_person_means':float(np.std([r[name] for r in pp],ddof=1)) for name in METRICS}))
    index={(r['subject'],r['method'],r['day'],r['variant']):r for r in people}
    accuracy={m:np.asarray([index[(s,m,2,'raw')]['accuracy'] for s in EVALUATION]) for m in METHOD_CONFIG}
    a,b,c,d=(accuracy[m] for m in 'ABCD')
    differences=[('C-A',c-a),('D-B',d-b),('B-A',b-a),('D-C',d-c),('(D-C)-(B-A)',(d-c)-(b-a))]
    contrasts=[dict(contrast=name,primary=(i==0),day=2,endpoint='native decision accuracy',**paired_difference_pp(100*delta)) for i,(name,delta) in enumerate(differences)]
    adjusted=holm5([r['p_two_sided'] for r in contrasts])
    for r,p in zip(contrasts,adjusted):r.update(p_Holm5=float(p),significant_Holm_005=bool(p<.05),Holm_family_size=5)
    return people,summary,contrasts

def atomic_json(path,value):
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,path)

def write_csv(path,rows):
    require(bool(rows),'Cannot write empty table')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def engine_array_sha(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256()
    for block in (str(a.dtype).encode('ascii'),json.dumps(list(a.shape)).encode('ascii'),a.tobytes()):
        h.update(len(block).to_bytes(8,'big'));h.update(block)
    return h.hexdigest()

def expected_ids():
    cvs={f'cv_source_hold{s:03d}_{m}' for s in SOURCES for m in METHOD_CONFIG}
    cvt={f'cv_target_hold{s:03d}_{m}' for s in SOURCES for m in METHOD_CONFIG}
    ms={f'main_source_{m}_seed{k}' for m in METHOD_CONFIG for k in SEEDS}
    mt={f'main_target_subject{s:03d}_{m}_seed{k}' for s in EVALUATION for m in METHOD_CONFIG for k in SEEDS}
    cc={f'cv_classical_hold{s:03d}_{m}' for s in SOURCES for m in CLASSICAL}
    mc={f'main_classical_subject{s:03d}_{m}' for s in EVALUATION for m in CLASSICAL}
    return dict(records=cvs|cvt|ms|mt,classical=cc|mc,temperatures=mt|mc,cv_validation=cvt|cc,
                test_records={f'{fid}_day{d}' for fid in mt|mc for d in DAYS},
                cv_source=cvs,cv_target=cvt,main_source=ms,main_target=mt,cv_classical=cc,main_classical=mc)

def verify_map(root,mapping,label,cache=None):
    require(isinstance(mapping,dict) and bool(mapping),'Empty/missing hash manifest '+label)
    for name,digest in mapping.items():
        if cache is None or name not in cache:
            require(sha(local(root,name))==digest,'Artifact hash mismatch '+label+'/'+name)
        else:require(cache[name]==digest,'Conflicting artifact hashes '+name)
        if cache is not None:cache[name]=digest

def verify_protocol(path,digest,root):
    require(sha(path)==digest,'Protocol SHA256 mismatch')
    p=read(path)
    require(p['status']=='FROZEN_BEFORE_CAMPAIGN','Protocol not frozen before campaign')
    require(p['groups']['source']==SOURCES and p['groups']['evaluation']==EVALUATION,'Participant partition changed')
    require(p['counts']==COUNTS,'Prespecified counts changed')
    counts={}
    for key in ('code_sha256','data_sha256','qa_sha256'):
        verify_map(root,p[key],key);counts[key]=len(p[key])
    require('analysis/final_analysis.py' in p['code_sha256'],'Analyzer must be frozen protocol code')
    require(len(p['protected_files'])==12,'Expected12 protected IEEE originals/copies')
    for key in ('protected_files','old_scientific_protection'):
        verify_map(root.parent,p[key],key);counts[key]=len(p[key])
    require(bool(p['readonly_day1_inputs']),'Day1 external input bindings absent')
    for name,expected in p['readonly_day1_inputs'].items():
        path=Path(name) if Path(name).is_absolute() else local(root.parent,name)
        require(path.resolve().is_relative_to(root.parent.resolve()) and sha(path)==expected,'External readonlyDay1 input changed '+name)
    counts['readonly_day1_inputs']=len(p['readonly_day1_inputs'])
    p['_audit_file_counts']=counts
    return p

def prediction_file(path,output=None):
    with np.load(path,allow_pickle=False) as z:
        keys={'scores','y_int64','pred_int64','source_indices_zero_based'}
        if output in ('CE','WALSH'):keys|={'logits_float32','probabilities_float32'}
        require(set(z.files)==keys,'Prediction NPZ schema mismatch '+str(path))
        a={key:z[key] for key in z.files}
    scores=checked_scores(a['scores']);y=labels(a['y_int64'],len(scores));pred=a['pred_int64'];indices=a['source_indices_zero_based']
    require(pred.dtype==np.int64 and pred.shape==y.shape and np.isin(pred,[0,1]).all(),'Invalid native decisions')
    require(indices.dtype==np.int64 and indices.shape==y.shape and len(np.unique(indices))==len(indices),'Invalid original recording indices')
    if output in ('CE','WALSH'):
        expected,native=native_scores(a['logits_float32'],a['probabilities_float32'],output)
        require(np.allclose(scores,expected,rtol=0,atol=1e-12) and np.array_equal(pred,native),'Native output/class score/decision mismatch')
    else:require(np.array_equal(pred,scores.argmax(1)),'Classical score/decision mismatch')
    return a

def verify_temperature(value,scores,y):
    require(value['source']=='reserved20Day1' and value['n_calibration']==20 and value['calibration_class_counts']==[10,10],'Wrong calibration source/budget')
    labels(y,20,10);checked_scores(scores)
    require(value['scores_sha256']==array_sha(scores) and value['labels_sha256']==array_sha(y),'Temperature fit array hashes changed')
    require(value['bounds']==[1.,100.] and value['optimizer_success'] is True and value['no_test_labels_used'] is True,'Wrong temperature constraints')
    T=float(value['temperature']);require(1.<=T<=100. and np.isfinite(T),'Temperature outside frozen bounds')
    require(value['at_lower_bound']==(T==1.) and value['at_upper_bound']==(T==100.),'Wrong temperature boundary flags')
    pred=scores.argmax(1).astype(np.int64)
    before=probability_metrics(scores,y,pred,1.)['nll'];after=probability_metrics(scores,y,pred,T)['nll']
    require(np.isclose(value['nll_before'],before,rtol=1e-10,atol=1e-11) and np.isclose(value['nll_after'],after,rtol=1e-10,atol=1e-11),'Temperature calibration NLL mismatch')
    require(after<=before+1e-10,'Temperature worsens reserved-data NLL against included T1 candidate')
    return T

def verify_original_labels(p,root,truth):
    path=Path(p['events_metadata_path']).resolve()
    require(path.is_relative_to(root.parent.resolve()),'Original events ZIP escapes project')
    require(sha(path)==truth['events_zip_sha256'],'Original event metadata ZIP hash mismatch')
    with zipfile.ZipFile(path) as z:
        for subject in SOURCES+EVALUATION:
            item=truth['day1'][str(subject)]
            rows=list(csv.reader(io.StringIO(z.read(f'events/sub-{subject:03d}_ses-01_task_motorimagery_events.tsv').decode('utf-8-sig')),delimiter='\t'))
            if rows and rows[0] and rows[0][0]=='onset':rows.pop(0)
            original=np.asarray([int(float(r[5]))-1 for r in rows if r],np.int64)
            ty=labels(np.asarray(item['train_y'],np.int64),50,25)
            cy=labels(np.asarray(item['calibration_y'],np.int64),20,10)
            ti,ci=np.asarray(item['train_indices'],np.int64),np.asarray(item['calibration_indices'],np.int64)
            require(ti.shape==(50,) and ci.shape==(20,) and len(set(ti.tolist()+ci.tolist()))==70,'Day1 originalrow partition overlaps/duplicates')
            require(np.all((ti>=0)&(ti<len(original))) and np.all((ci>=0)&(ci<len(original))),'Day1 row index outofrange')
            require(np.array_equal(original[ti],ty) and np.array_equal(original[ci],cy),'Day1 labels differ from independent originalevents')
            if subject in SOURCES:
                sy=labels(np.asarray(item['source70_y'],np.int64),70,35);si=np.asarray(item['source70_indices'],np.int64)
                require(si.shape==(70,) and set(si.tolist())==set(ti.tolist()+ci.tolist()) and np.array_equal(original[si],sy),'Source70 originalindices/labels mismatch')
        for subject in EVALUATION:
            for day in DAYS:
                key=f'subject_{subject:03d}_day{day}';item=truth['later'][key]
                rows=list(csv.reader(io.StringIO(z.read(f'events/sub-{subject:03d}_ses-{day:02d}_task_motorimagery_events.tsv').decode('utf-8-sig')),delimiter='\t'))
                if rows and rows[0] and rows[0][0]=='onset':
                    require(rows.pop(0)==['onset','duration','trial_type','response_time','sample','value'],'Unexpected event header')
                y=[int(float(r[5]))-1 for r in rows if r]
                require(y==item['y'] and item['source_indices_zero_based']==list(range(len(y))),'Independent event labels/all-trial recording order mismatch '+key)

def verify_complete_artifacts(p,ps,root):
    ids=expected_ids()
    # Do not open ANY outcome arrays/metric values until all expected identities
    # and all model/test byte manifests have passed the complete-case gate.
    for folder in ('records','classical','temperatures','cv_validation','test_records'):
        actual={path.stem for path in (root/'campaign'/folder).glob('*.json')}
        require(actual==ids[folder],'Incomplete/unexpected '+folder)
    require({path.stem for path in (root/'campaign/predictions').glob('*.npz')}==ids['test_records'],'Incomplete/unexpected test prediction arrays')
    freeze_path=root/'campaign/MODEL_FREEZE-V1.0.json';freeze=read(freeze_path);fs=sha(freeze_path)
    gate=read(root/'campaign/LATER_DAYS_GATE-V1.0.json');completion=read(root/'campaign/TEST_COMPLETION-V1.0.json')
    require(freeze['protocol_sha256']==gate['protocol_sha256']==completion['protocol_sha256']==ps,'Global receipt protocol mismatch')
    require(freeze['counts']==gate['counts']==COUNTS,'Incomplete fit gate counts')
    require(freeze['all_fit_artifacts_complete'] is True,'Fit/temperature freeze not complete')
    require(freeze['no_target_later_day_open_by_runner_before_freeze'] is True,'Required noearlylaterday declaration absent')
    require(gate['freeze_sha256']==gate['model_freeze_sha256']==completion['freeze_sha256']==fs,'Model freeze/test binding mismatch')
    require(completion['tests']==1320,'Final test completion count differs')
    ft,gt,ct=(moment(v) for v in (freeze['created_trt'],gate['opened_trt'],completion['completed_trt']))
    require(ft<=gt<=ct,'Freeze/gate/completion chronological order invalid')
    frozen={};tested={};verify_map(root,freeze['files'],'model freeze',frozen);verify_map(root,completion['files'],'test completion',tested)
    def record(folder,fid,test=False):
        rel=f'campaign/{folder}/{fid}.json';mapping=tested if test else frozen
        require(rel in mapping and sha(root/rel)==mapping[rel],'Record missing from global manifest '+rel)
        r=read(root/rel);require(r['protocol_sha256']==ps,'Record protocol mismatch '+fid)
        for name,digest in r.get('files',{}).items():require(mapping.get(name)==digest,'Record child unbound '+name)
        return r
    truth_path=root/'data/LABELS_AND_INDICES-V1.0.json'
    require(truth_path.relative_to(root).as_posix() in p['data_sha256'],'Truth label manifest must be protocol-bound')
    truth=read(truth_path);verify_original_labels(p,root,truth)
    require(set(truth['later'])=={f'subject_{s:03d}_day{d}' for s in EVALUATION for d in DAYS},'Later label identities differ')
    bindings_path=root/'data/INPUT_BINDINGS-V1.0.json'
    require(bindings_path.relative_to(root).as_posix() in p['data_sha256'],'Input bindings must be protocol-bound')
    bindings=read(bindings_path)
    require(set(bindings['later'])==set(truth['later']),'Later input bindings contain missing/extrapeople/days')
    for key,item in bindings['later'].items():
        truth_later=truth['later'][key]
        require(item['sha256']==truth_later['npz_sha256'] and item['metadata_sha256']==truth_later['metadata_sha256'],'Source label/input hash binding differs')
        for pathkey,hashkey in [('path','sha256'),('metadata_path','metadata_sha256')]:
            external=Path(item[pathkey]).resolve()
            require(external.is_relative_to(root.parent.resolve()) and sha(external)==item[hashkey],'Readonly later input bytes changed '+key+'/'+pathkey)
    neural={};preprocessors={};prefreeze_union={};costs=[];maxfit=None
    for fid in sorted(ids['records']):
        r=record('records',fid);tr=r['train_record'];m=r['condition'];neural[fid]=r
        require(m in METHOD_CONFIG and r['id']==fid and r['fit_complete'] is True,'Unknown/incomplete neural method/identity')
        architecture,output,nfeatures=METHOD_CONFIG[m]
        require(r['architecture']==tr['architecture']==architecture and r['output']==tr['output']==output,'Neural architecture/output mismatch')
        s,k=r['subject'],r['seed']
        if fid in ids['cv_source']:
            expected=f'cv_source_hold{s:03d}_{m}';people=[a for a in SOURCES if a!=s];n=630;init=510000+s;batch=610000+s
            require(r['stage']=='cv' and r['role']=='source' and s in SOURCES and k==1,'CV source metadata mismatch')
        elif fid in ids['cv_target']:
            expected=f'cv_target_hold{s:03d}_{m}';n=50;init=710000+s;batch=810000+s
            require(r['stage']=='cv' and r['role']=='target' and s in SOURCES and k==1,'CV target metadata mismatch')
        elif fid in ids['main_source']:
            expected=f'main_source_{m}_seed{k}';people=SOURCES;n=700;init=110000+k;batch=310000+k
            require(r['stage']=='main' and r['role']=='source' and s is None and k in SEEDS,'Main source metadata mismatch')
        else:
            expected=f'main_target_subject{s:03d}_{m}_seed{k}';n=50;init=210000+k;batch=410000+k
            require(r['stage']=='main' and r['role']=='target' and s in EVALUATION and k in SEEDS,'Main target metadata mismatch')
        require(fid==expected and tr['fit_trials']==n and tr['seed_init']==init and tr['seed_batch']==batch,'Neural fit identifier/budget/seeds mismatch')
        require(tr['blocks_completed']==10 and tr['epochs_per_block']==30 and tr['epochs_completed']==300 and tr['fit_prediction_passes']==10,'Not fixed300 neural epochs')
        updates={50:4,630:40,700:44}[n]
        require(tr['optimizer_updates_per_epoch']==updates and tr['optimizer_steps_per_parameter']==updates*300,'Adam update budget mismatch')
        require(tr['optimizer']=='Adam' and tr['learning_rate']==.001 and tr['batch_size']==16 and tr['fresh_optimizer_state'] is True,'Adam/configuration differs')
        require(tr['campaign_eligible'] is True and tr['engineering_only'] is False and tr['engineering_label'] is None and tr['test_arrays_accepted'] is False and tr['prediction_metrics_computed'] is False,'Pilot/test contamination flag')
        require(tr['feature_dimensions']==nfeatures and tr['output_width']==(2 if output=='CE' else 16),'Neural feature/output dimensions differ')
        require(tr['input_shape']==([29,512] if architecture=='EEGNet' else [9,29,512]) and tr['channels']==29 and tr['time_samples']==512,'Neural input dimensions differ')
        require(tr['loss']==('cross_entropy' if output=='CE' else 'binary_cross_entropy_with_logits') and tr['hard_targets_unchanged'] is True,'Training loss/target differs')
        for name in ('finite_gradients_every_optimizer_step','parameters_and_buffers_finite_every_block','fit_outputs_finite_in_unit_interval_every_block'):
            require(tr[name] is True,'Training numerical audit absent '+name)
        for name in ('checkpoint_path','preprocessor_path','fit_outputs_path'):require(r[name] in r['files'],'Unbound neural artifact '+name)
        pre=read(local(root,r['preprocessor_path']))
        preprocessors[fid]=pre
        individual_pre=list(pre.values()) if r['role']=='source' else [pre]
        for value in individual_pre:
            require(value['epsilon']==1e-6 and value['quantile']==.999 and value['method']=='linear','Preprocessing constants differ')
            arrays=[np.asarray(value[key],np.float64) for key in ('mean','denominator','clip_bounds')]
            require(all(a.shape==(1,29,1) and np.isfinite(a).all() for a in arrays) and (arrays[1]>0).all() and (arrays[2]>=0).all(),'Invalid preprocessing parameters')
        if r['role']=='source':
            require(set(pre)=={str(a) for a in people},'Source preprocessor/person pool mismatch')
            require(all(x['fit_n']==70 for x in pre.values()),'Source preprocessing not fit only70/person')
            expected_y=np.concatenate([np.asarray(truth['day1'][str(a)]['source70_y'],np.int64) for a in people])
        else:
            require(pre['fit_n']==50,'Target preprocessor fit budget differs')
            expected_y=np.asarray(truth['day1'][str(s)]['train_y'],np.int64)
        with np.load(local(root,r['fit_outputs_path']),allow_pickle=False) as z:
            require(set(z.files)=={'features_fit','logits_fit','P_fit','Y_used_last','y_fit'},'Neural fit NPZ schema differs')
            features,logits,prob,target,y=(z[name] for name in ('features_fit','logits_fit','P_fit','Y_used_last','y_fit'))
        require(features.dtype==np.float32 and features.shape==(n,nfeatures) and np.isfinite(features).all(),'Fit feature matrix invalid')
        require(np.array_equal(labels(y,n),expected_y),'Training class label order differs')
        native_scores(logits,prob,output)
        hard=y if output=='CE' else np.asarray([[1.]*16,[1.,0.]*8],np.float32)[y]
        require(np.array_equal(target,hard) and tr['hard_targets_sha256']==engine_array_sha(hard),'Training hard targets/hash differ')
        t=moment(r['completed_trt']);require(t<=ft,'Neural fit completed afterfreeze');maxfit=t if maxfit is None or t>maxfit else maxfit
        prefreeze_union[f'campaign/records/{fid}.json']=frozen[f'campaign/records/{fid}.json'];prefreeze_union.update(r['files'])
        costs.append(dict(fit_id=fid,stage=r['stage'],role=r['role'],condition=m,subject=s,seed=k,fit_trials=n,epochs=300,train_seconds=tr['train_seconds']))
    for fid,r in neural.items():
        tr=r['train_record'];m=r['condition']
        if r['role']=='target':
            source=f"cv_source_hold{r['subject']:03d}_{m}" if r['stage']=='cv' else f"main_source_{m}_seed{r['seed']}"
            final=neural[source]['train_record']['final_state_sha256']
            require(r['source_fit_id']==source and tr['mode']=='DIRECT' and tr['source_initial_state_sha256']==tr['initial_training_state_sha256']==tr['source_state_after_sha256']==final and tr['source_state_unchanged'] is True,'DIRECT fullsource state transfer mismatch')
        else:
            require(r['source_fit_id'] is None and tr['mode']=='FIXED' and tr['source_initial_state_sha256'] is None and tr['initial_fresh_state_sha256']==tr['initial_training_state_sha256'],'Source unexpectedly has transferred state')
        partner=m.translate(str.maketrans({'A':'B','B':'A','C':'D','D':'C'}))
        partner_id=fid.replace('_'+m+'_','_'+partner+'_') if '_'+m+'_' in fid else fid[:-1]+partner
        other=neural[partner_id]['train_record']
        require(tr['initial_fresh_backbone_sha256']==other['initial_fresh_backbone_sha256'] and tr['batch_generator_initial_sha256']==other['batch_generator_initial_sha256'],'CE/Walsh backbone initialization/batch seeds not paired')
        require(preprocessors[fid]==preprocessors[partner_id],'CE/Walsh preprocessing differs within pairedarchitecture')
        for other_m in METHOD_CONFIG:
            other_id=fid.replace('_'+m+'_','_'+other_m+'_') if '_'+m+'_' in fid else fid[:-1]+other_m
            require(preprocessors[fid]==preprocessors[other_id],'Preprocessing must be identical acrossallfour methods')
    classical={}
    for fid in sorted(ids['classical']):
        r=record('classical',fid);classical[fid]=r;s=r['subject'];m=r['condition']
        expected=f'cv_classical_hold{s:03d}_{m}' if fid in ids['cv_classical'] else f'main_classical_subject{s:03d}_{m}'
        require(fid==r['id']==expected and m in CLASSICAL and r['fit_trials']==50 and r['seed']==0,'Classical identity/budget differs')
        require((r['stage']=='cv' and s in SOURCES) if fid in ids['cv_classical'] else (r['stage']=='main' and s in EVALUATION),'Classical stage/participant differs')
        require(r['feature_count']==(6 if m=='SINGLE_CSP_LDA' else 54),'Classical feature count differs')
        require(r['architecture']=='CSP' and r['output']=='LDA','Classical architecture/output differs')
        cfg=r['configuration']
        require(cfg['condition']==m and cfg['feature_count']==r['feature_count'] and cfg['fit_trials']==50 and cfg['class_counts']==[25,25] and cfg['no_feature_selection'] is True and cfg['no_test_fit_or_refit'] is True,'Classical configuration/budget differs')
        require(cfg['CSP']['n_components']==6 and cfg['CSP']['component_order']=='alternate' and cfg['CSP']['reg']=='ledoit_wolf' and cfg['CSP']['rank']=='full' and cfg['CSP']['norm_trace'] is False,'Classical CSP differs')
        require(cfg['LDA']=={'solver':'lsqr','shrinkage':'auto','priors':[.5,.5]},'Classical LDA differs')
        if m=='SINGLE_CSP_LDA':require(cfg['filtering']['band_hz']==[8.,30.] and cfg['filtering']['fs']==128. and cfg['filtering']['padlen']==27,'Singleband filtering differs')
        else:require(cfg['filtering']['bands_hz']==[[float(lo),float(lo+4)] for lo in range(4,40,4)] and cfg['filtering']['fs']==128. and cfg['filtering']['padlen']==27 and cfg['n_bands']==9,'Shared9band filtering differs')
        for name in ('head_path','preprocessor_path'):require(r[name] in r['files'],'Classical child artifact unbound')
        head=joblib.load(local(root,r['head_path']))
        require(head['schema']=='shu-fixed-classical-pipeline-v1' and head['condition']==m and head['configuration']==cfg and head['fit_n']==50 and head['feature_count']==r['feature_count'],'Frozenclassical head/configuration mismatch')
        train_y=np.asarray(truth['day1'][str(s)]['train_y'],np.int64)
        require(head['fit_y_sha256']==array_sha(train_y),'Classical head notfit onexactfirstday50 labels')
        lda=head['pipeline']['lda']
        require(lda._study_fit_n==50 and lda._study_feature_count==r['feature_count'] and lda._study_fit_y_sha256==array_sha(train_y) and lda.solver=='lsqr' and lda.shrinkage=='auto' and np.array_equal(lda.priors_,[.5,.5]),'Frozen classical LDA fitbudget/labels/configuration mismatch')
        csps=[head['pipeline']['csp']] if m=='SINGLE_CSP_LDA' else head['pipeline']['csps']
        require(len(csps)==(1 if m=='SINGLE_CSP_LDA' else 9) and all(c.n_components==6 and c.component_order=='alternate' for c in csps),'Frozen classical CSP filtercount/order mismatch')
        require(read(local(root,r['preprocessor_path']))['fit_n']==50,'Classical preprocessing used otherthan50')
        require(moment(r['completed_trt'])<=ft,'Classical fitted afterfreeze')
        prefreeze_union[f'campaign/classical/{fid}.json']=frozen[f'campaign/classical/{fid}.json'];prefreeze_union.update(r['files'])
        costs.append(dict(fit_id=fid,stage=r['stage'],role='target',condition=m,subject=s,seed=0,fit_trials=50,epochs=None,train_seconds=r['train_seconds']))
    temperatures={};temperature_rows=[]
    for fid in sorted(ids['temperatures']):
        r=record('temperatures',fid);fr=neural[fid] if fid in neural else classical[fid]
        require(r['fit_id']==fid and r['subject']==fr['subject'] and r['condition']==fr['condition'] and r['seed']==fr['seed'],'Temperature/fit identity mismatch')
        require(r['trained_on']==50 and r['temperature_only_on_reserved']==20 and r['no_later_day_open'] is True,'Temperature information boundary differs')
        require(r['calibration_path'] in r['files'] and moment(r['completed_trt'])<=ft,'Calibration unbound orfitafterfreeze')
        a=prediction_file(local(root,r['calibration_path']),fr['output'] if fid in neural else None)
        truth1=truth['day1'][str(fr['subject'])]
        require(np.array_equal(a['y_int64'],truth1['calibration_y']) and np.array_equal(a['source_indices_zero_based'],truth1['calibration_indices']),'Calibration not exact reserved20 originalrows')
        temperatures[fid]=verify_temperature(r['temperature'],a['scores'],a['y_int64'])
        temperature_rows.append(dict(fit_id=fid,subject=fr['subject'],condition=fr['condition'],seed=fr['seed'],**r['temperature']))
        prefreeze_union[f'campaign/temperatures/{fid}.json']=frozen[f'campaign/temperatures/{fid}.json'];prefreeze_union.update(r['files'])
    cvrows=[]
    for fid in sorted(ids['cv_validation']):
        r=record('cv_validation',fid);fr=neural[fid] if fid in neural else classical[fid]
        require(r['fit_id']==fid and r['no_temperature_fitted'] is True and r['predictions_path'] in r['files'],'SourceCV validation schema differs')
        require(r['source_person_first_day_only'] is True and fr['stage']=='cv' and fr['subject'] in SOURCES and moment(r['completed_trt'])<=ft,'SourceCV metadata/time differs')
        a=prediction_file(local(root,r['predictions_path']),fr['output'] if fid in neural else None)
        truth1=truth['day1'][str(fr['subject'])]
        require(np.array_equal(a['y_int64'],truth1['calibration_y']) and np.array_equal(a['source_indices_zero_based'],truth1['calibration_indices']),'CV validation not sourcefirstday reserved20')
        actual=probability_metrics(a['scores'],a['y_int64'],a['pred_int64'])
        close_metrics(r['metrics'],actual,fid+'/CV');cvrows.append(dict(subject=fr['subject'],condition=fr['condition'],**actual))
        prefreeze_union[f'campaign/cv_validation/{fid}.json']=frozen[f'campaign/cv_validation/{fid}.json'];prefreeze_union.update(r['files'])
    require(prefreeze_union==freeze['files'],'Global freeze contains missing/unexpected fitted artifacts')
    rows=[];test_union={}
    for tid in sorted(ids['test_records']):
        r=record('test_records',tid,True);fid=r['fit_id'];s,d,m,k=r['subject'],r['day'],r['condition'],r['seed']
        require(r['id']==tid==f'{fid}_day{d}' and s in EVALUATION and d in DAYS and r['freeze_sha256']==fs and r['original_model_state_unchanged'] is True,'Wrong test identity/freeze/state')
        fr=neural[fid] if fid in ids['main_target'] else classical[fid]
        require(fr['stage']=='main' and fr['subject']==s and fr['condition']==m and fr['seed']==k and r['architecture']==fr['architecture'] and r['output']==fr['output'],'Test/fit identity mismatch')
        require(r['predictions_path'] in r['files'] and r['predictions_sha256']==r['files'][r['predictions_path']],'Test prediction path/hash mismatch')
        a=prediction_file(local(root,r['predictions_path']),fr['output'] if fid in neural else None)
        target=truth['later'][f'subject_{s:03d}_day{d}']
        require(np.array_equal(a['y_int64'],target['y']) and np.array_equal(a['source_indices_zero_based'],target['source_indices_zero_based']) and r['n']==len(target['y']),'Laterlabels/recordingindices notallsourceevents')
        require(r['temperature']==temperatures[fid] and gt<=moment(r['completed_trt'])<=ct,'Test useschangedtemperature ortimeoutsidegate')
        require(r['data_metadata']['npz_sha256']==target['npz_sha256'],'Test sourcearray hash notoriginalboundinput')
        variants={'raw':1.,'calibrated':temperatures[fid]}
        if m in ('B','D'):variants['T8']=8.
        require(set(r['metrics'])==set(variants),'Unexpected probability variants')
        for variant,T in variants.items():
            actual=probability_metrics(a['scores'],a['y_int64'],a['pred_int64'],T)
            close_metrics(r['metrics'][variant],actual,tid+'/'+variant)
            require(actual['accuracy']==r['metrics']['raw']['accuracy'],'Temperature changednative accuracy')
            rows.append(dict(test_id=tid,fit_id=fid,subject=s,method=m,seed=k,day=d,variant=variant,temperature=T,**actual))
        test_union[f'campaign/test_records/{tid}.json']=tested[f'campaign/test_records/{tid}.json'];test_union.update(r['files'])
    require(test_union==completion['files'],'Unexpected/missing final test manifest artifacts')
    audit=dict(status='ALL_COMPLETE_PROVENANCE_AND_METRICS_VERIFIED',neural_fits=400,cv_neural_fits=80,main_neural_fits=320,
        classical_fits=50,temperature_records=330,source_CV_validation_records=60,tests=1320,probability_rows=len(rows),
        model_freeze_artifacts=len(frozen),final_test_artifacts=len(tested),protocol_file_counts=p['_audit_file_counts'],
        EEG_values_or_native_models_or_checkpoints_deserialized=False,trusted_hash_verified_classical_heads_deserialized=True,
        readonly_later_source_arrays_and_metadata_hashed=120,source_target_fullstate_joins_verified=340,
        fresh_optimizer_and300epochs_verified=True,independent_original_eventlabels_verified=True,
        latest_fit_completed=maxfit.isoformat(),model_freeze_created=freeze['created_trt'],gate_opened=gate['opened_trt'],tests_completed=completion['completed_trt'],IEEE_files_unchanged=12)
    return rows,cvrows,temperature_rows,costs,fs,audit

def run_analysis(protocol_path,digest,root=ROOT):
    root=Path(root).resolve();p=verify_protocol(Path(protocol_path),digest,root)
    rows,cv,temperatures,costs,fs,audit=verify_complete_artifacts(p,digest,root)
    people,summary,contrasts=summarize(rows)
    result=dict(status='COMPLETE_ANALYSIS_VERIFIED',created_trt=datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),
        protocol_sha256=digest,model_freeze_sha256=fs,audit=audit,groups=p['groups'],evaluation_n=15,
        method_definitions={m:dict(architecture=v[0],output=v[1],features=v[2]) for m,v in METHOD_CONFIG.items()},
        classical_methods=list(CLASSICAL),summaries=summary,Day2_accuracy_Holm5=contrasts,
        inference='Exploratory15target-person means of5neural starts; classicalone deterministicfit/person/condition. Fixed10source-personpool.',
        primary='Day2 nativeaccuracy C-A; fouradditional prespecifiedcontrasts including pairedpersoninteraction formonefive-pvalue Holmfamily.',
        probability_mapping='CErawlogits; Walsh Bernoulli scores fromfloat64sigmoidwith1e-7bitclip; stablelogsumexpclassNLLwithoutclassprobabilityclipping.',
        decision='CErawlogitsargmax; Walshretainedfloat32sigmoid->Euclideancode; classicalscoreargmax; posteriorargmaxbeforeexp.',
        calibration='One boundedT[1,100] on20reservedDay1records only; all330temperatures frozenbeforelaterdays; originaldecisions unchanged.',
        T8='Fixed structuralWalshclassscore divisionby8, descriptiveonly andnotfitted.',
        uncertainty='Pointwise95%pairedt intervals are not multiplicityadjusted; fiveDay2nativeaccuracy pvalues Holmcorrected; otherdays/metrics descriptive.',
        limitations=['SHU andLee results alreadyseen; exploratorydevelopment, notnewuntouchedindependentvalidation.',
            'Current50train/20calibration models differfromprior70trainmodels; no pooledorpairedoldmodelclaims.',
            'No forgettingprevention experiment oruniversalsuperiority claim.',
            'Only15targetpeople; uncertainty conditionalonfixed10personsourcepool.',
            'Small20trialcalibration cannotguarantee reliability underlaterdayshift.',
            'FBCNet usesdeclaredSOSzerophasefilteradaptation andpureevalconstraints; notbyteidenticalauthorpipeline.',
            'Classicalsingle/bankCSP aredifferentfeaturepipelines; theircomparison isnotonlyclassifierheadsubstitution.'])
    out=root/'final_analysis';out.mkdir(parents=True,exist_ok=True)
    tables={'model_day_metrics-V1.0.csv':rows,'participant_means-V1.0.csv':people,'condition_summaries-V1.0.csv':summary,
            'Day2_Holm5-V1.0.csv':contrasts,'source_CV_validation-V1.0.csv':cv,'temperature_audit-V1.0.csv':temperatures,'training_costs-V1.0.csv':costs}
    for name,data in tables.items():write_csv(out/name,data)
    atomic_json(out/'FBCNET_RESULTS-V1.0.json',result)
    lines=['SHU FBCNET COMPARISON: COMPLETE EXPLORATORY RESULTS',
           '15 target people; five neural random starts averaged per person; classical one deterministic fit.',
           '50 first-day training trials and20 reserved first-day calibration trials. All later-day trials retained.',
           'A: EEGNet/CE; B: EEGNet/Walsh; C: FBCNet/CE; D: FBCNet/Walsh.',
           'Day2 accuracy contrasts below form exactly one five-test Holm family.',
           'Intervals are pointwise95% paired-person t intervals, not multiplicity-adjusted intervals.','',
           'DAY2 METHOD TABLE: accuracy(%), NLL, ECE15, summed Brier (raw -> calibrated)']
    for method in METHODS:
        raw=next(r for r in summary if r['method']==method and r['day']==2 and r['variant']=='raw')
        cal=next(r for r in summary if r['method']==method and r['day']==2 and r['variant']=='calibrated')
        lines.append(f"{method}: accuracy {100*raw['accuracy']:.4f}; NLL {raw['nll']:.6f} -> {cal['nll']:.6f}; ECE15 {raw['ece15']:.6f} -> {cal['ece15']:.6f}; Brier {raw['brier']:.6f} -> {cal['brier']:.6f}")
    lines+=['','DAY2 NATIVE-ACCURACY CONTRASTS (percentage points)']
    for contrast in contrasts:
        lo,hi=contrast['pointwise_95ci_pp']
        lines.append(f"{contrast['contrast']}: delta {contrast['delta_mean_pp']:.6f}; pointwise95%CI [{lo:.6f}, {hi:.6f}]; Holm5 p={contrast['p_Holm5']:.8g}; n=15")
    lines+=['','SHU was previously examined: these are exploratory development results, not untouched external validation.',
            'Temperature affects confidence only and does not change retained native decisions.',
            'The fixed10-person source pool and small20-trial calibration reserve limit generalization claims.',
            'IEEE Access submission files were hash-verified unchanged.']
    (out/'RESULTS_README-V1.0.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(out/'ANALYSIS_COMPLETION-V1.0.json',dict(status='COMPLETE_ANALYSIS_VERIFIED',protocol_sha256=digest,
        created_trt=datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(),files={str(path.relative_to(root).as_posix()):sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!='ANALYSIS_COMPLETION-V1.0.json'}))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();result=run_analysis(Path(args.protocol),args.protocol_sha256)
    print(json.dumps(dict(status=result['status'],audit=result['audit']),ensure_ascii=True))

if __name__=='__main__':main()
