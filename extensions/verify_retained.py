#!/usr/bin/env python3
"""Portable verification of completed extension predictions; no EEG, Torch or training.

Legacy Lee class NLL keeps its historical1e-12 final probability clipping.
Later calibration/SHU NLL is stable and unclipped at the class level; Walsh
bit1e-7 clipping is historical and is not removed by this verifier.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile,collections
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
def require(b,msg):
 if not b:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def close(a,b,label,atol=2e-9):require(np.isclose(a,b,atol=atol,rtol=0),f'{label}: {a!r} != {b!r}')
def ece(prob,y,posterior_pred=None):
 conf=prob.max(1);ok=(prob.argmax(1) if posterior_pred is None else posterior_pred)==y
 edges=np.linspace(0,1,16);total=0.
 for i in range(15):
  mask=((conf>=edges[i]) if i==0 else (conf>edges[i]))&(conf<=edges[i+1])
  if mask.any():total+=mask.mean()*abs(ok[mask].mean()-conf[mask].mean())
 return float(total)
def metrics(scores,y,pred,T=1):
 s=np.asarray(scores,dtype=np.float64)/T;m=s.max(1,keepdims=True)
 lp=s-(m+np.log(np.exp(s-m).sum(1,keepdims=True)));prob=np.exp(lp);pp=scores.argmax(1)
 return dict(accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean([(pred[y==c]==c).mean() for c in (0,1)])),
 recall_left=float((pred[y==0]==0).mean()),recall_right=float((pred[y==1]==1).mean()),nll=float(-lp[np.arange(len(y)),y].mean()),
 nll_legacy_clipped=float(-np.log(np.clip(prob[np.arange(len(y)),y],1e-12,1)).mean()),ece15=ece(prob,y,pp),
 brier=float(((prob-np.eye(2)[y])**2).sum(1).mean()),posterior_accuracy=float((pp==y).mean()),
 decoder_posterior_disagreement_fraction=float((pp!=pred).mean()),mean_max_probability=float(prob.max(1).mean())),prob
def checked_arrays(a,keys):
 require(set(a.files)==set(keys),'Unexpected prediction archive members')
 for k in a.files:require(a[k].dtype.kind in 'fiu' and np.isfinite(a[k]).all(),'Nonfinite/non-numeric archive member '+k)

def array_sha(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode('ascii'));h.update(json.dumps(list(a.shape),separators=(',',':')).encode('ascii'));h.update(a.tobytes());return h.hexdigest()

def verify_auxiliary(z):
 a=json.loads(z.read('AUX_INDEX.json'));sources=(21,6,10,18,19,25,8,7,15,12);targets=(22,14,17,4,1,16,2,5,11,20,9,24,3,23,13)
 expected={'shu_neural_fits':400,'shu_classical_fits':50,'shu_temperatures':330,'shu_source_validation':60,'lee_classical':54}
 require(a['expected_counts']==expected and {k:len(a[k]) for k in expected}==expected,'Complete auxiliary counts')
 neural={r['id']:r for r in a['shu_neural_fits']};require(len(neural)==400,'Unique neural fits')
 ids={f'cv_{role}_hold{s:03d}_{c}'for s in sources for c in 'ABCD'for role in ('source','target')}
 ids|={f'main_source_{c}_seed{k}'for c in 'ABCD'for k in range(1,6)}
 ids|={f'main_target_subject{s:03d}_{c}_seed{k}'for s in targets for c in 'ABCD'for k in range(1,6)}
 require(set(neural)==ids,'Exact400 neural identities including source/CV')
 for r in neural.values():
  tr=r['train_record'];n=50 if r['role']=='target'else (630 if r['stage']=='cv'else 700)
  require(tr['fit_trials']==n and tr['epochs_completed']==300 and tr['blocks_completed']==10 and tr['optimizer_steps_per_parameter']=={50:1200,630:12000,700:13200}[n],'Fixed neural training budgets')
  require(tr['campaign_eligible']and not tr['engineering_only']and tr['fresh_optimizer_state'],'Scientific fresh-Adam fit scope')
  require(not tr['test_arrays_accepted']and tr['hard_targets_unchanged'],'No test/training target adaptation')
  require((r['architecture'],r['output'])=={'A':('EEGNet','CE'),'B':('EEGNet','WALSH'),'C':('FBCNet','CE'),'D':('FBCNet','WALSH')}[r['condition']],'Architecture/output binding')
  if r['role']=='target':
   source=neural[r['source_fit_id']];state=source['train_record']['final_state_sha256']
   require(source['role']=='source'and source['condition']==r['condition']and source['seed']==r['seed'],'Matched source transfer identity')
   require(tr['initial_training_state_sha256']==tr['source_initial_state_sha256']==tr['source_state_after_sha256']==state and tr['source_state_unchanged'],'Full-state transfer provenance')
 classical={r['id']:r for r in a['shu_classical_fits']};cids={f'cv_classical_hold{s:03d}_{c}'for s in sources for c in ('SINGLE_CSP_LDA','BANK_CSP_LDA')}|{f'main_classical_subject{s:03d}_{c}'for s in targets for c in ('SINGLE_CSP_LDA','BANK_CSP_LDA')}
 require(set(classical)==cids and len(classical)==50,'Exact50 classical fit identities')
 for r in classical.values():require(r['fit_trials']==50 and r['feature_count']==(6 if r['condition']=='SINGLE_CSP_LDA'else 54),'Classical50 trials/six or54 features')
 tids={f'main_target_subject{s:03d}_{c}_seed{k}'for s in targets for c in 'ABCD'for k in range(1,6)}|{f'main_classical_subject{s:03d}_{c}'for s in targets for c in ('SINGLE_CSP_LDA','BANK_CSP_LDA')}
 temp={r['id']:r for r in a['shu_temperatures']};require(set(temp)==tids and len(temp)==330,'Exact330 heldout temperatures')
 for r in temp.values():
  raw=z.read(r['calibration_file']);require(sha(raw)==r['calibration_sha256'],'Calibration member SHA')
  with np.load(io.BytesIO(raw),allow_pickle=False)as d:
   keys=['scores','y_int64','pred_int64','source_indices_zero_based']
   if r['condition']in 'ABCD':keys+=['logits_float32','probabilities_float32']
   checked_arrays(d,keys);scores=d['scores'];y=d['y_int64'];pred=d['pred_int64'];t=r['temperature'];T=t['temperature']
   require(y.dtype==np.int64 and y.shape==(20,)and np.bincount(y,minlength=2).tolist()==[10,10]and scores.shape==(20,2),'Heldout20 calibration budget/class order')
   require(r['trained_on']==50 and r['temperature_only_on_reserved']==20 and r['no_later_day_open']and t['no_test_labels_used'],'Training/calibration/test separation')
   require(t['bounds']==[1.,100.]and 1<=T<=100 and t['n_calibration']==20,'Temperature positive prescribed bounds')
   require(array_sha(scores)==t['scores_sha256']and array_sha(y)==t['labels_sha256'],'Temperature input-array hashes')
   require(t['at_lower_bound']==bool(abs(T-1)<1e-6)and t['at_upper_bound']==bool(abs(T-100)<1e-4),'Temperature boundary flags')
   for field,scale in [('nll_before',1),('nll_after',T)]:close(metrics(scores,y,pred,scale)[0]['nll'],t[field],'Reserved20 '+field)
   require(t['nll_after']<=t['nll_before']+1e-12,'T1 included in calibration candidates')
 cv={r['id']:r for r in a['shu_source_validation']};cv_ids={f'cv_target_hold{s:03d}_{c}'for s in sources for c in 'ABCD'}|{f'cv_classical_hold{s:03d}_{c}'for s in sources for c in ('SINGLE_CSP_LDA','BANK_CSP_LDA')}
 require(set(cv)==cv_ids and len(cv)==60,'Exact60 source-only CV validation records')
 for r in cv.values():
  require(r['source_person_first_day_only']and r['no_temperature_fitted'],'Source CV no calibration fit on evaluated20')
  raw=z.read(r['prediction_file']);require(sha(raw)==r['prediction_sha256'],'SourceCV prediction SHA')
  with np.load(io.BytesIO(raw),allow_pickle=False)as d:
   keys=['scores','y_int64','pred_int64','source_indices_zero_based']
   if r['condition']in 'ABCD':keys+=['logits_float32','probabilities_float32']
   checked_arrays(d,keys);y=d['y_int64'];require(len(y)==20 and np.bincount(y,minlength=2).tolist()==[10,10],'SourceCV20 class budget')
   got,_=metrics(d['scores'],y,d['pred_int64'])
   for k in ('accuracy','balanced_accuracy','recall_left','recall_right','nll','ece15','brier','posterior_accuracy','decoder_posterior_disagreement_fraction','mean_max_probability'):close(got[k],r['metrics'][k],'SourceCV '+k)
 require({r['subject']for r in a['lee_classical']}==set(range(1,55)),'All54 Lee classical participants')
 for r in a['lee_classical']:
  require((r['n_train'],r['n_test'],r['n_features'])==(100,100,6),'Lee classical100/100/six feature budget')
  raw=z.read(r['prediction_file']);require(sha(raw)==r['prediction_sha256'],'Lee classical prediction SHA')
  with np.load(io.BytesIO(raw),allow_pickle=False)as d:
   checked_arrays(d,['P_test','y_test']);prob=d['P_test'];y=d['y_test'];require(y.dtype==np.int64 and y.shape==(100,)and prob.shape==(100,2),'Classical probability/label shapes')
   for k,v in {'acc_test':float((prob.argmax(1)==y).mean()),'nll_test':float(-np.log(np.clip(prob[np.arange(100),y],1e-12,1)).mean()),'ece_test':ece(prob,y)}.items():close(v,r[k],'Lee classical '+k)
 # Exact saved source-CV, temperature and400-fit cost rows are independently linked.
 for filename,rows,key in [('source_CV_validation-V1.0.csv',cv,'subject'),('temperature_audit-V1.0.csv',temp,'fit_id'),('training_costs-V1.0.csv',{**neural,**classical},'fit_id')]:
  table=list(csv.DictReader(io.StringIO(z.read('shu_2x2/tables/'+filename).decode('utf-8-sig'))));require(len(table)==len(rows),'Auxiliary table row count '+filename)
  for row in table:
   if filename.startswith('source_CV'):r=next(v for v in rows.values()if v['subject']==int(row['subject'])and v['condition']==row['condition']);values=r['metrics']
   elif filename.startswith('temperature'):r=rows[row['fit_id']];values=r['temperature']
   else:
    r=rows[row['fit_id']];values={**r['train_record'],'epochs':r['train_record']['epochs_completed']} if 'train_record'in r else r
   for k in ('accuracy','nll','ece15','brier','temperature','nll_before','nll_after','fit_trials','epochs','train_seconds'):
    if k in row and row[k]!='':close(float(row[k]),values[k],filename+'/'+k)
 return expected
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path);args=ap.parse_args()
 manifest=json.loads((ROOT/'extensions/RESULTS_MANIFEST.json').read_text(encoding='utf-8'))
 ar=ROOT/manifest['archive'];require(sha(ar.read_bytes())==manifest['archive_sha256'],'Archive SHA mismatch')
 counts=collections.Counter();seen=set();accgroups=collections.defaultdict(list);shu=collections.defaultdict(list)
 with zipfile.ZipFile(ar) as z:
  names=z.namelist();require(len(names)==len(set(names)) and set(names)==set(manifest['members']),'Missing/orphan/duplicate archive member')
  for n in names:
   require(not n.startswith(('/','\\')) and '..' not in Path(n).parts,'Unsafe archive path')
   require(sha(z.read(n))==manifest['members'][n],'Member SHA mismatch '+n)
  index=json.loads(z.read('RUN_INDEX.json'));runs=index['runs'];require(len(runs)==4020,'Expected4020 runs')
  require(index['W4']['results_included'] is False,'Pending W4 must not be counted as complete')
  expected={('lee54',f'subject_{s:02d}_train__{c}__seed{k}') for s in range(1,55) for c in ('S0','W0','W1','W2','W3') for k in range(1,6)}
  expected|={('lee_calibration80_20',f'subject_{s:02d}__{c}__seed{k}') for s in range(1,55) for c in ('S0','W0','W1','W2','W3') for k in range(1,6)}
  targets=(22,14,17,4,1,16,2,5,11,20,9,24,3,23,13)
  expected|={('shu_2x2',f'main_target_subject{s:03d}_{c}_seed{k}_day{d}') for s in targets for c in 'ABCD' for k in range(1,6) for d in (2,3,4,5)}
  expected|={('shu_2x2',f'main_classical_subject{s:03d}_{c}_day{d}') for s in targets for c in ('SINGLE_CSP_LDA','BANK_CSP_LDA') for d in (2,3,4,5)}
  require({(r['study'],r['id']) for r in runs}==expected,'Exact expected run identity set mismatch')
  C=np.stack([np.ones(16,np.float32),np.tile(np.array([1,0],np.float32),8)])
  for r in runs:
   ident=(r['study'],r['id']);require(ident not in seen,'Duplicate run identity');seen.add(ident);counts[r['study']]+=1
   raw=z.read(r['prediction_file']);require(sha(raw)==r['prediction_sha256'],'Run prediction SHA mismatch')
   with np.load(io.BytesIO(raw),allow_pickle=False) as a:
    st=r['study']
    if st=='lee54':
     checked_arrays(a,['y_test','P_test']);y=a['y_test'];net=a['P_test'];require(y.dtype==np.int64 and y.shape==(100,),'Lee labels')
     if r['condition']=='S0':
      post=net.astype(np.float64);pred=post.argmax(1)
     else:
      p=net.astype(np.float64);pred=((p[:,None,:]-C[None,:,:])**2).sum(-1).argmin(1)
      q=np.clip(p,1e-7,1-1e-7);sc=np.log(q)@C.T.astype(np.float64)+np.log(1-q)@(1-C.T.astype(np.float64));sc-=sc.max(1,keepdims=True)
      post=np.exp(sc);post/=post.sum(1,keepdims=True)
     got={'accuracy':float((pred==y).mean()),'nll':float(-np.log(np.clip(post[np.arange(100),y],1e-12,1)).mean()),'ece15':ece(post,y)}
     for k,v in got.items():close(v,r['metrics'][k],r['id']+'/'+k)
     accgroups[(r['condition'],r['subject'])].append(got)
    elif st=='lee_calibration80_20':
     checked_arrays(a,['y_test','P_network','logits_test','class_log_scores_test','posterior_raw','posterior_calibrated','original_decoder_predictions','temperature'])
     y=a['y_test'];scores=a['class_log_scores_test'];pred=a['original_decoder_predictions'];T=float(a['temperature']);close(T,r['temperature'],'calibration T')
     require(.05<=T<=100,'Calibration T bounds')
     for mode,t in [('raw',1),('calibrated',T)]:
      got,prob=metrics(scores,y,pred,t);saved=r['metrics'][mode]
      # Original calibration metrics define posterior decisions from exp(logp).
      got['ece15']=ece(prob,y);got['posterior_accuracy']=float((prob.argmax(1)==y).mean())
      mapping={'accuracy_original_decoder':'accuracy','accuracy_class_posterior':'posterior_accuracy','nll_unclipped':'nll','nll_legacy_clipped':'nll_legacy_clipped','ece_15_equal_width':'ece15','brier_sum_over_classes':'brier'}
      for k,gk in mapping.items():close(got[gk],saved[k],r['id']+'/'+mode+'/'+k)
      require(np.allclose(prob,a['posterior_'+mode],atol=2e-12,rtol=0),'Saved calibration posterior mismatch')
    elif st=='shu_2x2':
     k=['scores','y_int64','pred_int64','source_indices_zero_based']
     if r['condition'] in ('A','B','C','D'):k+=['logits_float32','probabilities_float32']
     checked_arrays(a,k);y=a['y_int64'];scores=a['scores'];pred=a['pred_int64']
     for mode,t in [('raw',1),('calibrated',r['temperature'])]+([('T8',8)] if 'T8' in r['metrics'] else []):
      got,_=metrics(scores,y,pred,t)
      for k in ('accuracy','balanced_accuracy','recall_left','recall_right','nll','ece15','brier','posterior_accuracy','decoder_posterior_disagreement_fraction','mean_max_probability'):close(got[k],r['metrics'][mode][k],r['id']+'/'+mode+'/'+k)
      shu[(r['condition'],r['day'],mode,r['subject'])].append(got)
    else:raise ValueError('Unrecognized study')
    require(y.dtype==np.int64 and y.ndim==1 and np.isin(y,[0,1]).all() and len(y)==r['n'],'Label identity/shape mismatch')
  require(dict(counts)==index['expected_counts'],'Study counts mismatch')
  auxiliary=verify_auxiliary(z)
  # Independently recover participant seed means and the published condition tables.
  table=list(csv.DictReader(io.StringIO(z.read('lee54/tables/condition_summary.csv').decode('utf-8-sig'))))
  for row in table:
   for field,metric,scale in [('accuracy_mean_percent','accuracy',100),('nll_mean','nll',1),('ece_mean','ece15',1)]:
    per=[np.mean([v[metric] for v in accgroups[(row['condition'],s)]]) for s in range(1,55)]
    close(float(np.mean(per))*scale,float(row[field]),'Lee condition '+field)
  table=list(csv.DictReader(io.StringIO(z.read('shu_2x2/tables/condition_summaries-V1.0.csv').decode('utf-8-sig'))))
  for row in table:
   if row['day'] not in ('2','3','4','5'):continue
   groups=[v for k,v in shu.items() if k[:3]==(row['method'],int(row['day']),row['variant'])]
   require(len(groups)==15,'SHU participant mean count')
   for k in ('accuracy','nll','ece15','brier'):
    close(float(np.mean([np.mean([v[k] for v in g]) for g in groups])),float(row[k]),'SHU condition '+k)
 result={'status':'PASS_PORTABLE_RETAINED_EXTENSION_VERIFICATION','prediction_files':sum(counts.values()),'counts':dict(counts),'auxiliary_counts':auxiliary,'raw_EEG_reads':0,'checkpoint_reads':0,'training_or_inference':0,'W4':'PENDING_NOT_INCLUDED','scope':'Hashes, exact identities, native accuracy/probability metrics, Lee/SHU condition means;400 neural training budgets/source inheritance,50 classical budgets,330 reserved20 T inputs/NLL,60 sourceCV metrics and54 Lee classical predictions. Optimizer optimum and historical inferential tests are retained but not rerun by this NumPy-only verifier.'}
 if args.report:
  require(not args.report.exists(),'Choose new report path');args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
