"""One prospective W4 ablation, with a global all270-model Day2 gate."""
from pathlib import Path
import os
for _key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import contextlib,hashlib,json,shutil,sys,time,uuid
import numpy as np
import torch
import training
import lee_core as core
import lee_data as data

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
PROTOCOL=ROOT/'ANKRAJ_KAPI_PROTOKOLU-V1.0.json'
MODEL_FREEZE=ROOT/'campaign/MODEL_FREEZE-V1.0.json'
TEST_COMPLETION=ROOT/'campaign/TEST_COMPLETION-V1.0.json'
PEOPLE=tuple(range(1,55));SEEDS=tuple(range(1,6))
COUNTS=dict(fits=270,preprocessors=54,tests=270)
_BINDINGS=None;_PROTOCOL_OK=None;_MODEL_GATE=None;_ANALYSIS_GATE=False
_HASH_DEPTH=0;_GUARD=False


def require(v,m):
    if not bool(v):raise RuntimeError(m)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    global _HASH_DEPTH
    _HASH_DEPTH+=1
    try:
        h=hashlib.sha256()
        with Path(path).open('rb')as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    finally:_HASH_DEPTH-=1


def atomic_json(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
    if immutable and path.exists():require(path.read_bytes()==raw,'Immutable JSON differs '+str(path));return
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        for attempt in range(6):
            try:os.replace(temp,path);break
            except OSError as e:
                if not isinstance(e,PermissionError)and getattr(e,'winerror',None)not in(5,32,33):raise
                if attempt==5:raise RuntimeError('Atomic JSON replacement exhausted6 bounded attempts')from e
                time.sleep(.1*2**attempt)
    finally:
        if temp.exists():temp.unlink()


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.resolve().is_relative_to((ROOT/'campaign/predictions').resolve()):
        require(_MODEL_GATE is not None and MODEL_FREEZE.is_file()and sha(MODEL_FREEZE)==_MODEL_GATE[1],
                'Prediction writes require the complete model gate')
    require(not path.exists(),'Existing orphan array must be preserved before replacement')
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('xb')as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        for attempt in range(6):
            try:os.replace(temp,path);break
            except OSError as e:
                if not isinstance(e,PermissionError)and getattr(e,'winerror',None)not in(5,32,33):raise
                if attempt==5:raise RuntimeError('Atomic NPZ replacement exhausted6 bounded attempts')from e
                time.sleep(.1*2**attempt)
    finally:
        if temp.exists():temp.unlink()


def fid(s,seed):return f'subject_{s:02d}_train__W4__seed{seed}'
def identities():return {fid(s,k):(s,k)for s in PEOPLE for k in SEEDS}
def fit_paths(key):
    return dict(record=f'campaign/fit_records/{key}.json',checkpoint=f'campaign/checkpoints/{key}.ckpt',
                fit_outputs=f'campaign/fit_outputs/{key}.npz')
def prep_path(s):return f'data/preprocessors/subject_{s:02d}.json'
def test_paths(key):return dict(record=f'campaign/test_records/{key}_day2.json',predictions=f'campaign/predictions/{key}_day2.npz')


def local(relative):
    raw=Path(relative);require(not raw.is_absolute()and'..'not in raw.parts,'Only relative new-study artifacts')
    q=(ROOT/raw).resolve();require(q.is_relative_to(ROOT.resolve()),'Artifact path escaped new study');return q


def install_guard(b):
    global _GUARD
    if _GUARD:return
    train={Path(v['train']['path']).resolve()for v in b['subjects'].values()}
    test={Path(v['test']['path']).resolve()for v in b['subjects'].values()}
    historic={Path(v['prediction']['path']).resolve()for v in b['historical'].values()}
    readonly={Path(path).resolve()for path in b['readonly_sha256']}
    fitout={local(fit_paths(key)['fit_outputs'])for key in identities()}
    predictions={local(test_paths(key)['predictions'])for key in identities()}
    checkpoints={local(fit_paths(key)['checkpoint'])for key in identities()}
    def audit(event,args):
        if event!='open'or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        q=Path(os.fsdecode(args[0])).resolve();mode=args[1]if len(args)>1 else None
        flags=args[2]if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str)and any(c in mode for c in'wax+'))or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if q in readonly and writing:raise RuntimeError('Source/protected files are read-only')
        if not q.is_relative_to(PROJECT.resolve()):return
        if q.suffix.lower()not in('.npz','.npy','.mat','.pt','.ckpt'):return
        if _HASH_DEPTH>0 and(q in readonly or q in fitout or q in predictions or q in checkpoints):return
        if q in train:require(_PROTOCOL_OK is not None,'Training arrays forbidden before frozen protocol');return
        if q in test or q in predictions:
            require(_MODEL_GATE is not None and MODEL_FREEZE.is_file()and sha(MODEL_FREEZE)==_MODEL_GATE[1],
                    'Day2 arrays forbidden before all270 models freeze');return
        if q in historic:require(_ANALYSIS_GATE,'Historical result arrays wait for complete new540 test gate');return
        if q in fitout or q in checkpoints:require(_PROTOCOL_OK is not None,'New fit artifact before frozen protocol');return
        # Temporary owned files have a suffix after .npz/.ckpt and are handled by
        # their final canonical filename checks in atomic writers/loaders.
        raise RuntimeError('Numeric archive/checkpoint outside fixed study whitelist '+str(q))
    sys.addaudithook(audit);_GUARD=True


def read_bindings(p):
    q=ROOT/'data/SOURCE_BINDINGS-V1.0.json'
    require(sha(q)==p['source_bindings_sha256'],'Frozen source binding changed')
    b=read(q)
    require(set(b['subjects'])=={str(s)for s in PEOPLE}and
            set(b['historical'])=={f'subject_{s:02d}_train__{c}__seed{k}'
                for s in PEOPLE for k in SEEDS for c in('W1','W2')},'Exact54 subjects/540 historical comparators')
    require(len(b['protected_files'])==12,'Original IEEE12 protected')
    return b


def runtime():
    return dict(python=sys.version.split()[0],numpy=np.__version__,torch=str(torch.__version__),
                cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)if torch.cuda.is_available()else None,threads=2)


def validate_protocol(path,ps,check_runtime=False):
    global _BINDINGS,_PROTOCOL_OK
    require(Path(path).resolve()==PROTOCOL.resolve()and sha(path)==ps,'Exact frozen protocol path/SHA')
    p=read(path)
    require(p['status']=='FROZEN_BEFORE_ANCHOR_GATE_ABLATION'and p['counts']==COUNTS,'Wrong W4 status/counts')
    require(p.get('analysis',{}).get('contrasts')==['W2-W4','W4-W1'],'Fixed two-contrast family')
    for relative,h in p['code_sha256'].items():require(sha(local(relative))==h,'Frozen executable code changed '+relative)
    for relative,h in p.get('qa_sha256',{}).items():require(sha(local(relative))==h,'Frozen prelaunch QA changed')
    b=read_bindings(p);_BINDINGS=b;install_guard(b)
    for path,h in b['readonly_sha256'].items():require(sha(path)==h,'Bound source changed '+path)
    for path,h in b['protected_files'].items():require(sha(path)==h,'Original IEEE source changed '+path)
    if check_runtime:
        require(p['runtime']==runtime(),'No silent software/GPU substitution')
        require(str(core.setup_torch(2))=='cuda','Campaign requires fixed CUDA device')
    _PROTOCOL_OK=ps
    return p


def archive_orphan(path,reason):
    q=Path(path)
    if not q.exists():return
    require(q.resolve().is_relative_to(ROOT.resolve()),'Never archive old source files')
    h=sha(q);target=ROOT/'ops/orphans'/(q.stem[:48]+'-'+h[:16]+'.bin')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():require(target.read_bytes()==q.read_bytes(),'Orphan evidence collision');q.unlink()
    else:os.replace(q,target)
    atomic_json(target.with_suffix('.json'),dict(original_relative=str(q.relative_to(ROOT)),sha256=h,reason=reason),immutable=True)


def prepare_subject(p,ps,s):
    b=read_bindings(p);source=b['subjects'][str(s)]
    for leaf in source.values():require(sha(leaf['path'])==leaf['sha256'],'Prepared source bytes changed')
    meta=read(source['meta']['path'])
    require(meta.get('data_units',meta.get('units'))=='uV'and meta['subject']==s,'Original subject/unit contract')
    x,y=data.load_part(source['train']['path'],'train')
    require(x.shape==(100,62,512),'Exact100x62x512 prepared first-day input')
    mean,sd=data.zscore_fit(x)
    prep=dict(subject=s,protocol_sha256=ps,source_bindings_sha256=p['source_bindings_sha256'],
        train_source_sha256=source['train']['sha256'],meta_source_sha256=source['meta']['sha256'],
        fit_trials=100,input_unit='uV',mean=mean.tolist(),sd=sd.tolist(),
        recipe='Exact original float64 mean/populationSD+1e-6 over trials/time; float32 transformed input')
    atomic_json(local(prep_path(s)),prep,immutable=True)
    return data.zscore_apply(x,mean,sd),y


def fit_identity(s,seed,p,ps):
    b=read_bindings(p)
    return dict(study_id=training.STUDY_ID,scope='CAMPAIGN',fit_id=fid(s,seed),subject=s,seed=seed,
        condition='W4',rule=training.RULE,protocol_sha256=ps,source_bindings_sha256=p['source_bindings_sha256'],
        train_source_sha256=b['subjects'][str(s)]['train']['sha256'],preprocessor_sha256=sha(local(prep_path(s))))


def validate_fit_record(r,p,ps):
    key=r['id'];require(key in identities(),'Unknown W4 fit identity')
    s,seed=identities()[key]
    require(type(r['subject'])is int and r['subject']==s and type(r['seed'])is int and r['seed']==seed
        and r['fit_id']==key and r['condition']=='W4'and r['rule']==training.RULE
        and r['protocol_sha256']==ps and r['source_bindings_sha256']==p['source_bindings_sha256'], 'W4 caller provenance')
    require(r['campaign_eligible']is True and r['engineering_only']is False and r['epochs_completed']==300
        and r['next_phase']==10 and r['actual_update_calls']==2100 and r['terminal_model']is True
        and r['no_test_arrays_accepted']is True and r['phase_test_metrics_computed']is False,'Fixed terminal300 training')
    require(r['identity']['caller']==fit_identity(s,seed,p,ps),'Exact normalized/source fit identity')
    expected=fit_paths(key)
    require(set(r['files'])=={'checkpoint','fit_outputs'},'Exactly checkpoint and training output leaves')
    for name,leaf in r['files'].items():
        require(leaf['path']==expected[name]and sha(local(leaf['path']))==leaf['sha256'],'Fixed fit artifact path/SHA')
    return r


def model_files(p,ps):
    files={};b=read_bindings(p)
    for s in PEOPLE:
        q=local(prep_path(s));r=read(q)
        require(r['subject']==s and r['protocol_sha256']==ps and r['fit_trials']==100
            and r['source_bindings_sha256']==p['source_bindings_sha256']
            and r['train_source_sha256']==b['subjects'][str(s)]['train']['sha256']
            and r['meta_source_sha256']==b['subjects'][str(s)]['meta']['sha256'], 'First-day normalizer metadata')
        for name in('mean','sd'):
            v=np.asarray(r[name],dtype=np.float64)
            require(v.shape==(1,62,1)and np.isfinite(v).all()and(name!='sd'or np.all(v>0)), 'Normalizer shape/finite')
        files[prep_path(s)]=sha(q)
    for key in identities():
        relative=fit_paths(key)['record'];r=validate_fit_record(read(local(relative)),p,ps)
        files[relative]=sha(local(relative))
        for leaf in r['files'].values():files[leaf['path']]=leaf['sha256']
    require(len(files)==864,'Full864 model/normalizer artifact union')
    return files


def create_model_freeze(p,ps):
    if MODEL_FREEZE.exists():return validate_model_freeze(p,ps,MODEL_FREEZE)
    require(not list((ROOT/'campaign/test_records').glob('*.json'))and
            not list((ROOT/'campaign/predictions').glob('*.npz')),'No Day2 outcomes before model freeze')
    value=dict(status='ALL270_FITS_FROZEN_BEFORE_DAY2',protocol_sha256=ps,
        source_bindings_sha256=p['source_bindings_sha256'],counts=dict(fits=270,preprocessors=54,artifacts=864),
        fit_ids=sorted(identities()),files=model_files(p,ps),Day2_arrays_decoded_before_freeze=False)
    atomic_json(MODEL_FREEZE,value,immutable=True)
    return validate_model_freeze(p,ps,MODEL_FREEZE)


def validate_model_freeze(p,ps,path=MODEL_FREEZE):
    global _MODEL_GATE
    require(Path(path).resolve()==MODEL_FREEZE.resolve(),'Exact model freeze path')
    r=read(path)
    require(r['status']=='ALL270_FITS_FROZEN_BEFORE_DAY2'and r['protocol_sha256']==ps
        and r['source_bindings_sha256']==p['source_bindings_sha256']
        and r['counts']==dict(fits=270,preprocessors=54,artifacts=864)
        and r['fit_ids']==sorted(identities())and r['Day2_arrays_decoded_before_freeze']is False,'Complete model gate identity')
    require(r['files']==model_files(p,ps),'Complete immutable864 artifact union')
    _MODEL_GATE=(ps,sha(path));return r


def require_model_gate(p,ps):
    if _MODEL_GATE is None:validate_model_freeze(p,ps)
    require(_MODEL_GATE==(ps,sha(MODEL_FREEZE)),'Model gate changed')


def validate_test_receipt(r,p,ps,decode=True):
    key=r['fit_id'];require(key in identities(),'Unknown test fit identity')
    s,seed=identities()[key];paths=test_paths(key)
    require(r['id']==key+'_day2'and type(r['subject'])is int and r['subject']==s
        and type(r['seed'])is int and r['seed']==seed and r['condition']=='W4'
        and r['protocol_sha256']==ps and r['source_bindings_sha256']==p['source_bindings_sha256']
        and r['model_freeze_sha256']==sha(MODEL_FREEZE)
        and r['source_fit_sha256']==sha(local(fit_paths(key)['record'])),'Test identity/frozen model')
    require(r['predictions_path']==paths['predictions']and r['prediction_sha256']==sha(local(paths['predictions']))
        and r['n_test']==100 and r['final_test_evaluations']==1,'One fixed final100-trial test')
    require(r['evaluation']==dict(model_state_unchanged=True,BN_unchanged=True,RNG_unchanged=True,
                                  eval_modes_unchanged=True,no_test_statistics_or_gradients=True),'Pure terminal evaluation proof')
    if not decode:return r
    with np.load(local(paths['predictions']),allow_pickle=False)as z:
        require(set(z.files)=={'P_test_float32','y_test_int64'},'Exact new prediction archive members')
        P=z['P_test_float32'];y=z['y_test_int64']
    require(P.dtype==np.float32 and P.shape==(100,16)and np.isfinite(P).all()and np.all((P>=0)&(P<=1))
        and y.dtype==np.int64 and y.shape==(100,)and np.isin(y,[0,1]).all()
        and np.bincount(y,minlength=2).tolist()==[50,50], 'Saved Walsh output/labels')
    metrics=core.walsh_metrics(P,y,data.code_rows())
    for target,source in(('acc_test','acc'),('nll_test','nll'),('ece_test','ece'),('bce_test','bce')):
        require(r[target]==metrics[source],'Original float64 Walsh metric replay '+target)
    require(r['pred_test']==metrics['pred'],'Original float64 native decision replay')
    return r


def validate_test_manifest(p,ps,path=TEST_COMPLETION):
    global _ANALYSIS_GATE
    require_model_gate(p,ps);require(Path(path).resolve()==TEST_COMPLETION.resolve(),'Exact test completion path')
    r=read(path);require(r['status']=='TESTS_COMPLETE'and r['protocol_sha256']==ps
        and r['model_freeze_sha256']==sha(MODEL_FREEZE)and r['counts']==dict(tests=270,artifacts=540),
        'Complete540 test gate')
    files={};records=[]
    for key in identities():
        paths=test_paths(key);record=validate_test_receipt(read(local(paths['record'])),p,ps,decode=False)
        records.append(record);files[paths['record']]=sha(local(paths['record']))
        files[paths['predictions']]=record['prediction_sha256']
    require(len(files)==540 and r['files']==files,'All270 exact receipts/predictions before decoding')
    for record in records:validate_test_receipt(record,p,ps)
    _ANALYSIS_GATE=True
    return r


class Campaign:
    def __init__(self,p,ps,status=None):self.p=p;self.ps=ps;self.status=status or(lambda **kw:None)
    def train_all(self):
        for s in PEOPLE:
            x,y=prepare_subject(self.p,self.ps,s)
            for seed in SEEDS:
                key=fid(s,seed);paths=fit_paths(key)
                if local(paths['record']).exists():validate_fit_record(read(local(paths['record'])),self.p,self.ps);continue
                archive_orphan(local(paths['fit_outputs']),'Unreceipted training output, same fixed terminal fit')
                self.status(stage='TRAIN_W4',subject=s,seed=seed,fit_id=key,phase=0)
                def progress(phase,total,seconds):self.status(stage='TRAIN_W4',subject=s,seed=seed,fit_id=key,
                    phase=phase,total_phases=total,train_seconds=seconds)
                model,record,outputs=training.fit(x,y,fit_identity(s,seed,self.p,self.ps),device='cuda',
                    checkpoint_path=local(paths['checkpoint']),resume=local(paths['checkpoint']).exists(),progress=progress)
                save_npz(local(paths['fit_outputs']),**outputs)
                record.update(id=key,n_train=100,preprocessor_path=prep_path(s),
                    files={name:dict(path=paths[name],sha256=sha(local(paths[name])))for name in('checkpoint','fit_outputs')})
                atomic_json(local(paths['record']),record,immutable=True);validate_fit_record(record,self.p,self.ps)
                del model;torch.cuda.empty_cache()
                if(ROOT/'campaign/STOP_AFTER_CURRENT_FIT').exists():return False
        create_model_freeze(self.p,self.ps);return True
    def test_all(self):
        require_model_gate(self.p,self.ps);b=read_bindings(self.p)
        for s in PEOPLE:
            source=b['subjects'][str(s)]
            require(sha(source['test']['path'])==source['test']['sha256'],'Bound Day2 source changed before decoding')
            xraw,y=data.load_part(source['test']['path'],'test')
            require(xraw.shape==(100,62,512),'Exact prepared Day2 shape')
            prep=read(local(prep_path(s)))
            x=data.zscore_apply(xraw,np.asarray(prep['mean'],dtype=np.float64),np.asarray(prep['sd'],dtype=np.float64))
            for seed in SEEDS:
                key=fid(s,seed);paths=test_paths(key)
                if local(paths['record']).exists():validate_test_receipt(read(local(paths['record'])),self.p,self.ps);continue
                archive_orphan(local(paths['predictions']),'Unreceipted fixed terminal prediction')
                fit=validate_fit_record(read(local(fit_paths(key)['record'])),self.p,self.ps)
                cp=training.load_checkpoint(local(fit_paths(key)['checkpoint']),expected_identity=fit['identity']['caller'],
                    expected_sha256=fit['files']['checkpoint']['sha256'],require_complete=True)
                model=core.build_model('EEGNet',62,512,16,data.PARAMS['dropout']).to('cuda')
                model.load_state_dict(cp['state']['model'],strict=True);model.eval()
                require(training.model_hash(model)==fit['final_model_sha256'],'Frozen terminal model/fit receipt linkage')
                before=training.model_hash(model);rng=training.tree_hash(core._rng_state())
                modes={name:m.training for name,m in model.named_modules()}
                self.status(stage='TEST_DAY2',subject=s,seed=seed,fit_id=key)
                result,arrays=core.evaluate_final(model,x,y,walsh=True,C=data.code_rows(),dev=torch.device('cuda'))
                require(training.model_hash(model)==before and training.tree_hash(core._rng_state())==rng
                    and modes=={name:m.training for name,m in model.named_modules()},'Frozen eval model/BN/RNG/mode purity')
                save_npz(local(paths['predictions']),P_test_float32=arrays['P_test'],y_test_int64=arrays['y_test'])
                result.update(id=key+'_day2',fit_id=key,subject=s,seed=seed,condition='W4',n_test=100,
                    protocol_sha256=self.ps,source_bindings_sha256=self.p['source_bindings_sha256'],
                    model_freeze_sha256=sha(MODEL_FREEZE),source_fit_sha256=sha(local(fit_paths(key)['record'])),
                    predictions_path=paths['predictions'],prediction_sha256=sha(local(paths['predictions'])),
                    evaluation=dict(model_state_unchanged=True,BN_unchanged=True,RNG_unchanged=True,
                        eval_modes_unchanged=True,no_test_statistics_or_gradients=True),
                    decoder='Exact original float64 Walsh Euclidean decision; historical bit/class clipping retained')
                atomic_json(local(paths['record']),result,immutable=True);validate_test_receipt(result,self.p,self.ps)
                del cp,model;torch.cuda.empty_cache()
        files={}
        for key in identities():
            paths=test_paths(key);r=validate_test_receipt(read(local(paths['record'])),self.p,self.ps)
            files[paths['record']]=sha(local(paths['record']));files[paths['predictions']]=r['prediction_sha256']
        atomic_json(TEST_COMPLETION,dict(status='TESTS_COMPLETE',protocol_sha256=self.ps,
            model_freeze_sha256=sha(MODEL_FREEZE),counts=dict(tests=270,artifacts=540),files=files),immutable=True)
        return validate_test_manifest(self.p,self.ps)
    def run(self):
        if MODEL_FREEZE.exists():validate_model_freeze(self.p,self.ps)
        elif not self.train_all():return'STOPPED_AFTER_CURRENT_FIT'
        self.test_all();return'TESTS_COMPLETE'
