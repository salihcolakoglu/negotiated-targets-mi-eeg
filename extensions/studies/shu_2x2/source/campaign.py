"""400 fixed CNN fits, all classical/T freezes, then1320 later-day tests."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,copy,json,platform,subprocess,sys,time,traceback,uuid
import numpy as np
import torch,scipy,sklearn,mne,joblib
import models,training,filterbank,full_data as data,classical
import classifiers_metrics as cm

ROOT=Path(__file__).resolve().parents[1];TRT=timezone(timedelta(hours=3))
METHODS={'A':('EEGNet','CE'),'B':('EEGNet','WALSH'),'C':('FBCNet','CE'),'D':('FBCNet','WALSH')}


def stamp():return datetime.now(TRT).isoformat()
def read(p):return json.loads(Path(p).read_bytes())
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def sha(p):return data.sha(p)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(6):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.1*2**attempt)


def filemap(paths):return {rel(p):sha(p) for p in paths}


def checked(path,ps):
    r=read(path)
    if r['protocol_sha256']!=ps:raise RuntimeError('Record protocol changed:'+str(path))
    for name,digest in r.get('files',{}).items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Record artifact changed:'+name)
    return r


def valid_protocol(path,ps):
    if sha(path)!=ps:raise RuntimeError('Protocol hash changed')
    p=read(path)
    if p['status']!='FROZEN_BEFORE_CAMPAIGN' or p['counts']!=data.COUNTS:raise RuntimeError('Wrong frozen protocol')
    actual={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'sklearn':sklearn.__version__,'mne':mne.__version__,
            'python_executable':str(Path(sys.executable).resolve()),'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if actual!=p['environment']:raise RuntimeError('Runtime environment changed:'+str(actual))
    if not torch.cuda.is_available():raise RuntimeError('Do not silently change training device')
    for key in ('code_sha256','data_sha256','qa_sha256'):
        for name,digest in p[key].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Frozen file changed:'+name)
    for key in ('protected_files','old_scientific_protection','readonly_day1_inputs'):
        for name,digest in p[key].items():
            f=Path(name) if Path(name).is_absolute() else ROOT.parent/name
            if sha(f)!=digest:raise RuntimeError('Readonly file changed:'+name)
    if p['filterbank']!=filterbank.configuration():raise RuntimeError('Filter bank changed')
    return p


def model_from_record(rec,device):
    cp=training.load_checkpoint(ROOT/rec['checkpoint_path'])
    if cp['engineering_only'] or cp['completed_blocks']!=10:raise RuntimeError('Engineering/incomplete model in main campaign')
    m=models.build_model(rec['architecture'],rec['output']).to(device)
    m.load_state_dict(cp['model_state'],strict=True);m.eval()
    if training.state_hash(m)!=rec['train_record']['final_state_sha256']:raise RuntimeError('Model checkpoint mismatch')
    return m


def outputs(model,X,output,device):
    before=training.state_hash(model)
    logits,probs=models.predict_outputs(model,torch.from_numpy(np.ascontiguousarray(X)).to(device))
    if training.state_hash(model)!=before:raise RuntimeError('Inference changed weights/BN')
    if output=='CE':scores=logits.astype(np.float64);pred=scores.argmax(1).astype(np.int64)
    else:scores,pred=cm.walsh_scores(logits,probs)
    return scores,pred,{'logits_float32':logits,'probabilities_float32':probs}


class Campaign:
    def __init__(self,p,ps,path,resume=False):
        self.p,self.ps,self.path,self.resume=p,ps,Path(path),resume
        self.device=models.setup_torch(2);self.current=None
        for folder in ('records','classical','temperatures','cv_validation','checkpoints','preprocessors',
                       'fit_outputs','calibration','heads','test_records','predictions'):
            (ROOT/'campaign'/folder).mkdir(parents=True,exist_ok=True)
        self.neural_done=len(list((ROOT/'campaign/records').glob('*.json')))
        self.tests_done=len(list((ROOT/'campaign/test_records').glob('*.json')))
        for folder,ids in data.expected_ids().items():
            existing={f.stem for f in (ROOT/'campaign'/folder).glob('*.json')}
            if not existing.issubset(ids):raise RuntimeError('Unknown existing identities:'+folder)

    def status(self,stage,**extra):
        atomic_json(ROOT/'STATUS.json',{'time_trt':stamp(),'pid':os.getpid(),'stage':stage,
            'neural_fits_done':self.neural_done,'neural_fits_expected':400,'tests_done':self.tests_done,
            'tests_expected':1320,'classical_done':len(list((ROOT/'campaign/classical').glob('*.json'))),
            'temperature_done':len(list((ROOT/'campaign/temperatures').glob('*.json'))),'current':self.current,**extra})

    def source_arrays(self,people):
        xs,ys,states=[],[],{}
        for subject in people:
            x,y=data.load_source70(subject);st=data.fit_preprocessor(x)
            xs.append(data.apply_preprocessor(x,st));ys.append(y);states[str(subject)]=st
        return np.concatenate(xs),np.concatenate(ys),states

    def prepared(self,x):
        return {'EEGNet':np.ascontiguousarray(x,dtype=np.float32),
                'FBCNet':np.ascontiguousarray(filterbank.apply_bank(x),dtype=np.float32)}

    def fit(self,fid,X,y,stage,role,subject,condition,seed,init_seed,batch_seed,pre,source_id=None,initial=None):
        rp=ROOT/'campaign/records'/f'{fid}.json'
        if rp.exists():
            rec=checked(rp,self.ps)
            return model_from_record(rec,self.device),rec
        arch,output=METHODS[condition];self.current=fid;self.status('CV' if stage=='cv' else 'MAINFIT')
        cp=ROOT/'campaign/checkpoints'/f'{fid}.checkpoint'
        model,tr,arrays=training.fit_main(X,y,arch,output,init_seed,batch_seed,self.device,
            initial_state=initial,checkpoint_path=cp,resume=self.resume and cp.exists(),
            progress=lambda b,n,t:self.status('CV' if stage=='cv' else 'MAINFIT',block=b,blocks=n,fit_elapsed_seconds=t),
            run_fingerprint={'protocol_sha256':self.ps,'fit_id':fid})
        if not tr['campaign_eligible'] or tr['engineering_only']:raise RuntimeError('Engineering model rejected')
        pp=ROOT/'campaign/preprocessors'/f'{fid}.json';atomic_json(pp,pre)
        fp=ROOT/'campaign/fit_outputs'/f'{fid}.npz';fp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(fp,**arrays)
        rec={'id':fid,'protocol_sha256':self.ps,'completed_trt':stamp(),'stage':stage,'role':role,'subject':subject,
             'condition':condition,'architecture':arch,'output':output,'seed':seed,'source_fit_id':source_id,
             'train_record':tr,'fit_complete':True,'checkpoint_path':rel(cp),'preprocessor_path':rel(pp),
             'fit_outputs_path':rel(fp),'files':filemap([cp,pp,fp])}
        atomic_json(rp,rec);self.neural_done+=1
        print(json.dumps({'event':'fit_complete','id':fid,'done':self.neural_done,'expected':400,'time_trt':stamp()}),flush=True)
        if (ROOT/'STOP_AFTER_CURRENT_FIT').exists():raise RuntimeError('User STOP_AFTER_CURRENT_FIT requested; preserve checkpoint')
        return model,rec

    def save_cv(self,fid,subject,condition,scores,pred,vy,extras=None):
        rp=ROOT/'campaign/cv_validation'/f'{fid}.json'
        if rp.exists():checked(rp,self.ps);return
        vp=rp.with_suffix('.npz');vp.parent.mkdir(parents=True,exist_ok=True)
        truth=data.labels_manifest()['day1'][str(subject)]
        np.savez_compressed(vp,scores=scores,y_int64=vy,pred_int64=pred,
            source_indices_zero_based=np.asarray(truth['calibration_indices'],dtype=np.int64),**(extras or {}))
        atomic_json(rp,{'id':fid,'fit_id':fid,'protocol_sha256':self.ps,'completed_trt':stamp(),'subject':subject,
            'condition':condition,'source_person_first_day_only':True,'no_temperature_fitted':True,
            'metrics':cm.probability_metrics(scores,vy,pred,1.),'predictions_path':rel(vp),'files':filemap([vp])})

    def fit_classical(self,fid,subject,condition,stage,X,y,V,vy,pre):
        rp=ROOT/'campaign/classical'/f'{fid}.json'
        if rp.exists():
            rec=checked(rp,self.ps);head=joblib.load(ROOT/rec['head_path'])
            if stage=='cv':
                before=joblib.hash(head);scores=classical.predict_scores(head,V)
                if joblib.hash(head)!=before:raise RuntimeError('Classical resume scoring changed state')
                self.save_cv(fid,subject,condition,scores,scores.argmax(1).astype(np.int64),vy)
            return head,rec
        self.current=fid;self.status('CV' if stage=='cv' else 'MAINFIT')
        t=time.perf_counter();head=classical.fit_pipeline(X,y,condition);seconds=time.perf_counter()-t
        before=joblib.hash(head);scores=classical.predict_scores(head,V)
        if joblib.hash(head)!=before:raise RuntimeError('Classical scoring changed fitted model')
        hp=ROOT/'campaign/heads'/f'{fid}.joblib';hp.parent.mkdir(parents=True,exist_ok=True);joblib.dump(head,hp)
        pp=ROOT/'campaign/preprocessors'/f'{fid}.json';atomic_json(pp,pre)
        rec={'id':fid,'protocol_sha256':self.ps,'completed_trt':stamp(),'stage':stage,'role':'target',
             'subject':subject,'condition':condition,'architecture':'CSP','output':'LDA','seed':0,
             'fit_trials':50,'feature_count':6 if condition=='SINGLE_CSP_LDA' else 54,
             'configuration':classical.configuration(condition),'train_seconds':seconds,
             'head_path':rel(hp),'preprocessor_path':rel(pp),'fitted_state_joblib_hash':before,'files':filemap([hp,pp])}
        atomic_json(rp,rec)
        if stage=='cv':self.save_cv(fid,subject,condition,scores,scores.argmax(1).astype(np.int64),vy)
        return head,rec

    def cv(self):
        for hold in data.SOURCES:
            Xs,ys,states=self.source_arrays([s for s in data.SOURCES if s!=hold]);sx=self.prepared(Xs)
            X,y,V,vy=data.load_split(hold);pre=data.fit_preprocessor(X)
            Xt=data.apply_preprocessor(X,pre);Vt=data.apply_preprocessor(V,pre);tx=self.prepared(Xt);vx=self.prepared(Vt)
            for cond,(arch,out) in METHODS.items():
                sid=f'cv_source_hold{hold:03d}_{cond}'
                source,sr=self.fit(sid,sx[arch],ys,'cv','source',hold,cond,1,510000+hold,610000+hold,states)
                initial={k:v.detach().cpu().clone() for k,v in source.state_dict().items()};del source
                fid=f'cv_target_hold{hold:03d}_{cond}'
                model,rec=self.fit(fid,tx[arch],y,'cv','target',hold,cond,1,710000+hold,810000+hold,pre,sid,initial)
                sc,pred,extras=outputs(model,vx[arch],out,self.device);self.save_cv(fid,hold,cond,sc,pred,vy,extras)
                del model,initial
            for cond in data.CLASSICAL:
                self.fit_classical(f'cv_classical_hold{hold:03d}_{cond}',hold,cond,'cv',Xt,y,Vt,vy,pre)
            del Xs,sx,tx,vx;torch.cuda.empty_cache()

    def calibrate(self,rec,scores,pred,vy,extras=None):
        fid=rec['id'];tp=ROOT/'campaign/temperatures'/f'{fid}.json'
        if tp.exists():checked(tp,self.ps);return
        temp=cm.fit_temperature(scores,vy);cp=ROOT/'campaign/calibration'/f'{fid}.npz';cp.parent.mkdir(parents=True,exist_ok=True)
        truth=data.labels_manifest()['day1'][str(rec['subject'])]
        np.savez_compressed(cp,scores=scores,y_int64=vy,pred_int64=pred,
            source_indices_zero_based=np.asarray(truth['calibration_indices'],dtype=np.int64),**(extras or {}))
        atomic_json(tp,{'id':fid,'fit_id':fid,'protocol_sha256':self.ps,'completed_trt':stamp(),
            'subject':rec['subject'],'condition':rec['condition'],'architecture':rec['architecture'],'output':rec['output'],
            'seed':rec['seed'],'temperature':temp,'calibration_path':rel(cp),'trained_on':50,
            'temperature_only_on_reserved':20,'no_later_day_open':True,'files':filemap([cp])})

    def main_fits(self):
        Xs,ys,states=self.source_arrays(data.SOURCES);sx=self.prepared(Xs)
        for cond,(arch,out) in METHODS.items():
            for seed in range(1,6):
                sid=f'main_source_{cond}_seed{seed}'
                source,sr=self.fit(sid,sx[arch],ys,'main','source',None,cond,seed,110000+seed,310000+seed,states)
                initial={k:v.detach().cpu().clone() for k,v in source.state_dict().items()};del source
                for subject in data.TARGETS:
                    X,y,V,vy=data.load_split(subject);pre=data.fit_preprocessor(X)
                    Xt=data.apply_preprocessor(X,pre);Vt=data.apply_preprocessor(V,pre)
                    tx=(filterbank.apply_bank(Xt) if arch=='FBCNet' else Xt).astype(np.float32)
                    vx=(filterbank.apply_bank(Vt) if arch=='FBCNet' else Vt).astype(np.float32)
                    fid=f'main_target_subject{subject:03d}_{cond}_seed{seed}'
                    model,rec=self.fit(fid,tx,y,'main','target',subject,cond,seed,210000+seed,410000+seed,pre,sid,initial)
                    sc,pred,extras=outputs(model,vx,out,self.device);self.calibrate(rec,sc,pred,vy,extras)
                    del model
                del initial;torch.cuda.empty_cache()
        del sx,Xs;torch.cuda.empty_cache()
        for subject in data.TARGETS:
            X,y,V,vy=data.load_split(subject);pre=data.fit_preprocessor(X)
            Xt=data.apply_preprocessor(X,pre);Vt=data.apply_preprocessor(V,pre)
            for cond in data.CLASSICAL:
                fid=f'main_classical_subject{subject:03d}_{cond}'
                head,rec=self.fit_classical(fid,subject,cond,'main',Xt,y,Vt,vy,pre)
                before=joblib.hash(head);sc=classical.predict_scores(head,Vt)
                if joblib.hash(head)!=before:raise RuntimeError('Classical calibration changed state')
                self.calibrate(rec,sc,sc.argmax(1).astype(np.int64),vy)

    def freeze(self):
        valid_protocol(self.path,self.ps);fp=ROOT/'campaign/MODEL_FREEZE-V1.0.json'
        if fp.exists():data.validate_freeze(self.p,self.ps,fp);return fp
        files={}
        for folder,ids in data.expected_ids().items():
            if folder=='test_records':continue
            if {f.stem for f in (ROOT/'campaign'/folder).glob('*.json')}!=ids:raise RuntimeError('Missing fit identity:'+folder)
            for fid in sorted(ids):
                rp=ROOT/'campaign'/folder/(fid+'.json');r=checked(rp,self.ps)
                files.update(r.get('files',{}));files[rel(rp)]=sha(rp)
        if any(e['kind']=='later' for e in data.READ_EVENTS):raise RuntimeError('Later EEG was opened before freeze')
        atomic_json(fp,{'protocol_sha256':self.ps,'created_trt':stamp(),'counts':self.p['counts'],'files':files,
            'all_fit_artifacts_complete':True,'no_target_later_day_open_by_runner_before_freeze':True,
            'SHU_already_seen_exploratory_not_new_independent_validation':True})
        data.validate_freeze(self.p,self.ps,fp);return fp

    def write_test(self,rec,day,y,indices,meta,scores,pred,T,freeze_sha,extras=None):
        tid=f"{rec['id']}_day{day}";rp=ROOT/'campaign/test_records'/f'{tid}.json'
        if rp.exists():
            previous=checked(rp,self.ps)
            if previous['freeze_sha256']!=freeze_sha:raise RuntimeError('Test freeze changed')
            return
        pp=ROOT/'campaign/predictions'/f'{tid}.npz'
        pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,scores=np.asarray(scores,dtype=np.float64),y_int64=y.astype(np.int64),
            pred_int64=pred.astype(np.int64),source_indices_zero_based=indices.astype(np.int64),**(extras or {}))
        metrics={'raw':cm.probability_metrics(scores,y,pred,1.),'calibrated':cm.probability_metrics(scores,y,pred,T)}
        if rec['output']=='WALSH':metrics['T8']=cm.probability_metrics(scores,y,pred,8.)
        atomic_json(rp,{'id':tid,'fit_id':rec['id'],'protocol_sha256':self.ps,'freeze_sha256':freeze_sha,
            'subject':rec['subject'],'condition':rec['condition'],'architecture':rec['architecture'],'output':rec['output'],
            'seed':rec['seed'],'day':day,'n':len(y),'temperature':T,'metrics':metrics,'data_metadata':meta,
            'predictions_path':rel(pp),'predictions_sha256':sha(pp),'files':filemap([pp]),
            'original_model_state_unchanged':True,'completed_trt':stamp()})
        self.tests_done+=1;self.status('TEST')

    def tests(self,freeze_path):
        freeze_sha=sha(freeze_path);gp=ROOT/'campaign/LATER_DAYS_GATE-V1.0.json'
        if not gp.exists():atomic_json(gp,{'opened_trt':stamp(),'protocol_sha256':self.ps,'freeze_sha256':freeze_sha,
            'model_freeze_sha256':freeze_sha,'counts':self.p['counts'],'no_training_or_temperature_update_after_gate':True})
        data.authorize_later(self.p,self.ps,freeze_path)
        for subject in data.TARGETS:
            later={day:data.load_later(subject,day,self.p,self.ps,freeze_path) for day in (2,3,4,5)}
            for cond,(arch,out) in METHODS.items():
                for seed in range(1,6):
                    fid=f'main_target_subject{subject:03d}_{cond}_seed{seed}';r=checked(ROOT/'campaign/records'/f'{fid}.json',self.ps)
                    pre=read(ROOT/r['preprocessor_path']);temp=checked(ROOT/'campaign/temperatures'/f'{fid}.json',self.ps)['temperature']['temperature']
                    model=model_from_record(r,self.device);before=training.state_hash(model)
                    for day,(X,y,indices,meta) in later.items():
                        self.current=f'{fid}_day{day}';z=data.apply_preprocessor(X,pre)
                        prepared=(filterbank.apply_bank(z) if arch=='FBCNet' else z).astype(np.float32)
                        sc,pred,extras=outputs(model,prepared,out,self.device)
                        if training.state_hash(model)!=before:raise RuntimeError('Test mutated model')
                        self.write_test(r,day,y,indices,meta,sc,pred,temp,freeze_sha,extras)
                    del model
            for cond in data.CLASSICAL:
                fid=f'main_classical_subject{subject:03d}_{cond}';r=checked(ROOT/'campaign/classical'/f'{fid}.json',self.ps)
                pre=read(ROOT/r['preprocessor_path']);temp=checked(ROOT/'campaign/temperatures'/f'{fid}.json',self.ps)['temperature']['temperature']
                head=joblib.load(ROOT/r['head_path']);before=joblib.hash(head)
                for day,(X,y,indices,meta) in later.items():
                    sc=classical.predict_scores(head,data.apply_preprocessor(X,pre))
                    if joblib.hash(head)!=before:raise RuntimeError('Test mutated classical model')
                    self.write_test(r,day,y,indices,meta,sc,sc.argmax(1).astype(np.int64),temp,freeze_sha)
            torch.cuda.empty_cache()
        paths=list((ROOT/'campaign/test_records').glob('*.json'))
        if {f.stem for f in paths}!=data.expected_ids()['test_records']:raise RuntimeError('Test identities incomplete')
        files={}
        for rp in paths:
            r=checked(rp,self.ps);files.update(r['files']);files[rel(rp)]=sha(rp)
        atomic_json(ROOT/'campaign/TEST_COMPLETION-V1.0.json',{'protocol_sha256':self.ps,'freeze_sha256':freeze_sha,
            'tests':1320,'completed_trt':stamp(),'files':files})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['validate','run']);ap.add_argument('--protocol',required=True)
    ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    data.install_guard();p=valid_protocol(a.protocol,a.protocol_sha256)
    if a.command=='validate':print(json.dumps({'status':'PASS','counts':p['counts']}));return
    complete=ROOT/'ops/CAMPAIGN_COMPLETION-V1.0.json'
    if complete.exists():
        c=read(complete)
        if c['protocol_sha256']!=a.protocol_sha256 or c['final_result_sha256']!=sha(ROOT/'final_analysis/FBCNET_RESULTS-V1.0.json'):raise RuntimeError('Completion mismatch')
        print('ALREADY_COMPLETE_NO_NEW_TRAINING');return
    if not a.resume and any((ROOT/'campaign/records').glob('*.json')):raise RuntimeError('Existing campaign requires verified resume')
    lock=ROOT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started_trt':stamp(),'protocol_sha256':a.protocol_sha256},f)
    try:
        c=Campaign(p,a.protocol_sha256,a.protocol,a.resume);c.cv();c.main_fits();c.status('FREEZE')
        freeze=c.freeze();c.tests(freeze);c.status('ANALYSIS')
        subprocess.run([sys.executable,'-B',str(ROOT/'analysis/final_analysis.py'),'--protocol',a.protocol,
                        '--protocol-sha256',a.protocol_sha256],check=True,cwd=str(ROOT))
        atomic_json(complete,{'completed_trt':stamp(),'protocol_sha256':a.protocol_sha256,
            'neural_fits':400,'classical_fits':50,'temperature_values':330,'tests':1320,
            'final_result_sha256':sha(ROOT/'final_analysis/FBCNET_RESULTS-V1.0.json')})
        c.status('COMPLETE')
    except BaseException as exc:
        atomic_json(ROOT/'ops'/f'CAMPAIGN_FAILURE-{datetime.now(TRT).strftime("%Y%m%dT%H%M%S")}-V1.0.json',{
            'time_trt':stamp(),'pid':os.getpid(),'protocol_sha256':a.protocol_sha256,'exception':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
