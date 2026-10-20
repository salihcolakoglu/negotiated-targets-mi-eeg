"""Training-only first-day loaders and complete-model gate for later SHU EEG."""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OLD=PROJECT/'SHU_ONISLEME_VE_LDA_KARSILASTIRMA-V1.0'
SOURCES=[21,6,10,18,19,25,8,7,15,12]
TARGETS=[22,14,17,4,1,16,2,5,11,20,9,24,3,23,13]
CONDITIONS=('A','B','C','D')
CLASSICAL=('SINGLE_CSP_LDA','BANK_CSP_LDA')
COUNTS={'cv_neural_fits':80,'main_neural_fits':320,'classical_fits':50,
        'target_temperature_records':330,'cv_validation_records':60,'tests':1320}
_GATE_KEY=None
READ_EVENTS=[]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def bindings():return read(ROOT/'data/INPUT_BINDINGS-V1.0.json')
def labels_manifest():return read(ROOT/'data/LABELS_AND_INDICES-V1.0.json')


def expected_ids():
    cvsource={f'cv_source_hold{h:03d}_{c}' for h in SOURCES for c in CONDITIONS}
    cvtarget={f'cv_target_hold{h:03d}_{c}' for h in SOURCES for c in CONDITIONS}
    mainsource={f'main_source_{c}_seed{k}' for c in CONDITIONS for k in range(1,6)}
    maintarget={f'main_target_subject{s:03d}_{c}_seed{k}' for s in TARGETS for c in CONDITIONS for k in range(1,6)}
    cvclassic={f'cv_classical_hold{h:03d}_{c}' for h in SOURCES for c in CLASSICAL}
    mainclassic={f'main_classical_subject{s:03d}_{c}' for s in TARGETS for c in CLASSICAL}
    return {'records':cvsource|cvtarget|mainsource|maintarget,'classical':cvclassic|mainclassic,
            'temperatures':maintarget|mainclassic,'cv_validation':cvtarget|cvclassic,
            'test_records':{f'{f}_day{d}' for f in maintarget|mainclassic for d in (2,3,4,5)}}


def _arrays(x,y,n):
    if x.dtype!=np.float64 or x.shape!=(n,29,512) or not np.isfinite(x).all():raise ValueError('Invalid source-scaled EEG')
    if y.dtype!=np.int64 or y.shape!=(n,) or not np.isin(y,[0,1]).all():raise ValueError('Invalid labels')


def load_source70(subject):
    if subject not in SOURCES:raise PermissionError('Only source people have source70 access')
    b=bindings()['day1'][str(subject)]['source70'];path=Path(b['path'])
    if sha(path)!=b['sha256']:raise RuntimeError('Source70 byte hash mismatch')
    with np.load(path,allow_pickle=False) as d:x=d['X_fit'];y=d['y_fit'];indices=d['fit_indices_zero_based']
    _arrays(x,y,70);truth=labels_manifest()['day1'][str(subject)]
    if not np.array_equal(y,truth['source70_y']) or not np.array_equal(indices,truth['source70_indices']):raise RuntimeError('Source labels/indices mismatch')
    return x,y


def load_split(subject):
    if subject not in SOURCES+TARGETS:raise PermissionError('Unknown participant')
    b=bindings()['day1'][str(subject)]['split'];path=Path(b['path'])
    if sha(path)!=b['sha256']:raise RuntimeError('Split byte hash mismatch')
    with np.load(path,allow_pickle=False) as d:
        x,y,v,vy=d['X_train'],d['y_train'],d['X_calibration'],d['y_calibration']
        ix,iv=d['train_original_indices_zero_based'],d['calibration_original_indices_zero_based']
    _arrays(x,y,50);_arrays(v,vy,20);truth=labels_manifest()['day1'][str(subject)]
    if any(not np.array_equal(a,truth[k]) for a,k in [(y,'train_y'),(vy,'calibration_y'),(ix,'train_indices'),(iv,'calibration_indices')]):raise RuntimeError('First-day labels/indices mismatch')
    if np.bincount(y,minlength=2).tolist()!=[25,25] or np.bincount(vy,minlength=2).tolist()!=[10,10] or not set(ix).isdisjoint(iv):raise RuntimeError('50/20 information boundary violation')
    return x,y,v,vy


def fit_preprocessor(x):
    if x.dtype!=np.float64 or x.ndim!=3 or x.shape[1:]!=(29,512) or len(x) not in (50,70) or not np.isfinite(x).all():raise ValueError('Preprocessor fits only50/70finite training trials')
    mean=x.mean(axis=(0,2),keepdims=True);den=x.std(axis=(0,2),ddof=0,keepdims=True)+1e-6
    q=np.quantile(np.abs((x-mean)/den),.999,axis=(0,2),keepdims=True,method='linear')
    return {'mean':mean.tolist(),'denominator':den.tolist(),'clip_bounds':q.tolist(),'fit_n':len(x),'epsilon':1e-6,'quantile':.999,'method':'linear'}


def apply_preprocessor(x,p):
    if x.dtype!=np.float64 or x.ndim!=3 or x.shape[1:]!=(29,512) or not np.isfinite(x).all():raise ValueError('Invalid preprocessing EEG')
    mean,den,q=(np.asarray(p[k],dtype=np.float64) for k in ('mean','denominator','clip_bounds'))
    if p['fit_n'] not in (50,70) or p['quantile']!=.999 or p['epsilon']!=1e-6 or p['method']!='linear' or any(a.shape!=(1,29,1) or not np.isfinite(a).all() for a in (mean,den,q)) or np.any(den<=0) or np.any(q<0):raise ValueError('Invalid frozen train-only preprocessing')
    out=np.clip((x-mean)/den,-q,q)
    if not np.isfinite(out).all():raise FloatingPointError('Preprocessing overflow')
    return out


def validate_freeze(protocol,ps,freeze_path):
    if protocol['counts']!=COUNTS:raise RuntimeError('Frozen count contract changed')
    if sha(ROOT/'TAM_DENEY_PROTOKOLU-V1.0.json')!=ps:raise RuntimeError('Protocol changed')
    freeze=read(freeze_path)
    if freeze['protocol_sha256']!=ps or freeze['counts']!=COUNTS or freeze['all_fit_artifacts_complete'] is not True:raise RuntimeError('Invalid all-fit freeze')
    if not freeze['no_target_later_day_open_by_runner_before_freeze']:raise RuntimeError('Pre-freeze later EEG')
    union={};neural={}
    required={'records':('checkpoint_path','preprocessor_path','fit_outputs_path'),
              'classical':('head_path','preprocessor_path'),
              'temperatures':('calibration_path',),'cv_validation':('predictions_path',)}
    for folder,ids in expected_ids().items():
        if folder=='test_records':continue
        actual={f.stem for f in (ROOT/'campaign'/folder).glob('*.json')}
        if actual!=ids:raise RuntimeError('Missing/extra frozen identities:'+folder)
        for identity in sorted(ids):
            f=ROOT/'campaign'/folder/(identity+'.json');r=read(f)
            if r['protocol_sha256']!=ps:raise RuntimeError('Metadata protocol mismatch')
            if r.get('id',r.get('fit_id'))!=identity:raise RuntimeError('Metadata identity mismatch')
            paths={r.get(key) for key in required[folder]}
            if None in paths or set(r.get('files',{}))!=paths:raise RuntimeError('Required fitted artifacts missing:'+folder)
            if folder=='records':
                tr=r['train_record'];expected_n=50 if r['role']=='target' else (630 if r['stage']=='cv' else 700)
                expected_steps={50:1200,630:12000,700:13200}[expected_n]
                if (not r['fit_complete'] or not tr['campaign_eligible'] or tr['engineering_only']
                    or tr['fit_trials']!=expected_n or tr['epochs_completed']!=300 or tr['blocks_completed']!=10
                    or tr['optimizer_steps_per_parameter']!=expected_steps or not tr['fresh_optimizer_state']):
                    raise RuntimeError('Incomplete/engineering/wrong-budget fit')
                neural[identity]=r
            if folder=='classical' and (r['fit_trials']!=50 or r['feature_count'] not in (6,54)):
                raise RuntimeError('Wrong classical training budget')
            if folder=='temperatures':
                t=r['temperature']
                if r['trained_on']!=50 or r['temperature_only_on_reserved']!=20 or t['n_calibration']!=20 or t['bounds']!=[1.,100.] or not 1<=t['temperature']<=100:
                    raise RuntimeError('Wrong held-out calibration boundary')
            for rel,digest in r.get('files',{}).items():
                path=(ROOT/rel).resolve()
                if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest:raise RuntimeError('Frozen artifact mismatch:'+rel)
                union[rel]=digest
            union[f.relative_to(ROOT).as_posix()]=sha(f)
    for fid,r in neural.items():
        if r['role']=='target':
            source=neural.get(r['source_fit_id']);tr=r['train_record']
            if source is None or source['role']!='source' or source['condition']!=r['condition'] or source['seed']!=r['seed']:
                raise RuntimeError('Invalid own-source transfer identity')
            state=source['train_record']['final_state_sha256']
            if tr['initial_training_state_sha256']!=state or tr['source_initial_state_sha256']!=state or tr['source_state_after_sha256']!=state or not tr['source_state_unchanged']:
                raise RuntimeError('Full source state/BN transfer mismatch')
    if union!=freeze['files']:raise RuntimeError('Freeze not exact metadata/artifact union')
    for rel,digest in protocol['code_sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Scientific code changed')
    for rel,digest in protocol['protected_files'].items():
        if sha(PROJECT/rel)!=digest:raise RuntimeError('IEEE changed')
    return freeze


def authorize_later(protocol,ps,freeze_path):
    global _GATE_KEY
    f=validate_freeze(protocol,ps,freeze_path);fs=sha(freeze_path)
    g=read(ROOT/'campaign/LATER_DAYS_GATE-V1.0.json')
    if g['protocol_sha256']!=ps or g['freeze_sha256']!=fs or g['model_freeze_sha256']!=fs or g['counts']!=COUNTS:raise RuntimeError('Wrong later-day gate')
    if g['opened_trt']<f['created_trt']:raise RuntimeError('Gate predates model freeze')
    _GATE_KEY=(ps,fs)


def load_later(subject,day,protocol,ps,freeze_path):
    if subject not in TARGETS or day not in (2,3,4,5):raise PermissionError('Only evaluation later days allowed')
    if _GATE_KEY!=(ps,sha(freeze_path)):raise PermissionError('All models/T must freeze before later EEG')
    key=f'subject_{subject:03d}_day{day}';b=bindings()['later'][key];truth=labels_manifest()['later'][key]
    if sha(b['path'])!=b['sha256'] or sha(b['metadata_path'])!=b['metadata_sha256']:raise RuntimeError('Later data/meta mismatch')
    with np.load(b['path'],allow_pickle=False) as d:x=d['X_test'];y=d['y_test'];indices=d['source_indices_zero_based']
    _arrays(x,y,len(indices))
    if not np.array_equal(y,truth['y']) or not np.array_equal(indices,truth['source_indices_zero_based']):raise RuntimeError('Original events label/row order mismatch')
    return x,y,indices,{'old_npz_path':b['path'],'npz_sha256':b['sha256'],'metadata_path':b['metadata_path'],
                        'metadata_sha256':b['metadata_sha256'],'events_zip_sha256':labels_manifest()['events_zip_sha256'],
                        'subject':subject,'day':day,'all_eligible_trials_retained':True}


def install_guard():
    b=bindings();day1={os.path.normcase(str(Path(z['path']).resolve())) for v in b['day1'].values() for z in v.values()}
    later={os.path.normcase(str(Path(z['path']).resolve())) for z in b['later'].values()}
    own=os.path.normcase(str(ROOT.resolve()))+os.sep
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if Path(path).suffix.lower() not in ('.npz','.npy','.mat','.gdf','.edf'):return
        if path.startswith(own):return
        if path in day1:READ_EVENTS.append({'kind':'day1','path':path});return
        if path in later and _GATE_KEY is not None:READ_EVENTS.append({'kind':'later','path':path});return
        raise PermissionError('Frozen FBCNet EEG information boundary blocked '+path)
    sys.addaudithook(audit)
