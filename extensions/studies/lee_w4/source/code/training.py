"""Versioned phase checkpoint wrapper around the unchanged Lee training loop.

Only W4 (ungated, hard-label-anchored updates) is a new campaign condition.
No test array, label, checkpoint selection or budget override is accepted.
The private short generated-QA path cannot produce a campaign-eligible fit.
"""
from pathlib import Path
import copy,hashlib,io,json,os,struct,time,uuid
import numpy as np
import torch
import lee_core as core
import lee_data as data

STUDY_ID='LEE2019_ANKRAJ_KAPI_AYRISTIRMA-V1.0'
MAGIC=b'LEE2019_W4_UNGATED_ANCHORED_PHASE_CHECKPOINT_V1\x00'
SCHEMA='lee2019-w4-ungated-anchored-full-phase-state-v1'
RULE='neg_anchor_all'
PARAMS=dict(data.PARAMS)


def need(v,m):
    if not bool(v):raise RuntimeError(m)


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(value):
    digest=hashlib.sha256()
    def frame(raw):digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    def visit(v):
        if torch.is_tensor(v):
            frame(b'tensor');frame(str(v.dtype).encode());frame(str(tuple(v.shape)).encode())
            frame(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            frame(b'array');frame(str(v.dtype).encode());frame(str(v.shape).encode())
            frame(np.ascontiguousarray(v).tobytes())
        elif isinstance(v,dict):
            frame(b'dict')
            for k in sorted(v,key=lambda x:(str(type(x)),repr(x))):visit(k);visit(v[k])
        elif isinstance(v,(tuple,list)):
            frame(type(v).__name__.encode());frame(str(len(v)).encode())
            for x in v:visit(x)
        else:frame(type(v).__name__.encode());frame(repr(v).encode())
    visit(value);return digest.hexdigest()


def cpu_copy(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,np.ndarray):return v.copy()
    if isinstance(v,dict):return {k:cpu_copy(x)for k,x in v.items()}
    if isinstance(v,tuple):return tuple(cpu_copy(x)for x in v)
    if isinstance(v,list):return [cpu_copy(x)for x in v]
    return copy.deepcopy(v)


def model_hash(model):return tree_hash(model.state_dict())


def validate_identity(identity,engineering=False):
    c=json.loads(canonical(identity))
    need(c.get('study_id')==STUDY_ID and c.get('scope')==('SYNTHETIC_QA'if engineering else'CAMPAIGN'),
         'Wrong versioned study/scope')
    s=c.get('subject');seed=c.get('seed')
    need(type(s)is int and s in range(1,55)and type(seed)is int and seed in range(1,6),'Wrong person/seed')
    need(c.get('fit_id')==f'subject_{s:02d}_train__W4__seed{seed}'and c.get('condition')=='W4'
         and c.get('rule')==RULE,'Only fixed W4 identity is campaign eligible')
    for key in ('protocol_sha256','source_bindings_sha256','train_source_sha256','preprocessor_sha256'):
        v=c.get(key);need(isinstance(v,str)and len(v)==64 and all(x in'0123456789abcdef'for x in v),
                        'Missing fixed provenance '+key)
    return c


def _atomic_checkpoint(path,payload):
    stream=io.BytesIO();torch.save(payload,stream);body=stream.getvalue()
    raw=MAGIC+struct.pack('>Q',len(body))+hashlib.sha256(body).digest()+body
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        for attempt in range(6):
            try:os.replace(temp,path);break
            except OSError as e:
                if not isinstance(e,PermissionError)and getattr(e,'winerror',None)not in(5,32,33):raise
                if attempt==5:raise RuntimeError('Phase checkpoint replace failed after6 bounded attempts')from e
                time.sleep(.1*2**attempt)
    finally:
        if temp.exists():temp.unlink()


def load_checkpoint(path,*,expected_identity=None,expected_sha256=None,allow_engineering=False,require_complete=False):
    raw=Path(path).read_bytes()
    if expected_sha256 is not None:need(hashlib.sha256(raw).hexdigest()==expected_sha256,'Checkpoint full SHA changed')
    header=len(MAGIC)+40
    need(len(raw)>=header and raw[:len(MAGIC)]==MAGIC,'Reject legacy/other-study checkpoint')
    body=raw[header:]
    need(struct.unpack('>Q',raw[len(MAGIC):len(MAGIC)+8])[0]==len(body)and
         hashlib.sha256(body).digest()==raw[len(MAGIC)+8:header],'Checkpoint size/payloadSHA mismatch')
    cp=torch.load(io.BytesIO(body),map_location='cpu',weights_only=False)
    need(cp.get('schema')==SCHEMA and type(cp.get('engineering'))is bool and
         (not cp['engineering']or allow_engineering),'Wrong schema/engineering checkpoint')
    ident=validate_identity(cp['identity']['caller'],cp['engineering'])
    if expected_identity is not None:need(ident==expected_identity,'Wrong person/seed/rule/source/protocol checkpoint')
    need(cp['identity_canonical']==canonical(cp['identity'])and tree_hash(cp['state'])==cp['state_sha256'],
         'Immutable checkpoint state/identity changed')
    phases=cp['identity']['phases'];epochs=cp['identity']['epochs_per_phase'];st=cp['state']
    n=cp['identity']['input_shape'][0]
    need(cp['identity']['input_shape'][1:]==[62,512]and
         (cp['engineering']or(n==100 and phases==10 and epochs==30)), 'Wrong fixed dimensions/budget')
    need(type(st['next_phase'])is int and 1<=st['next_phase']<=phases and
         len(st['used_dist'])==len(st['used_gate'])==len(st['phase_seconds'])==st['next_phase'],
         'Incomplete phase/target history')
    need(all(v is None for v in st['used_gate'])and st['gate']is None,'W4 must not perform a correctness gate')
    need(set(st)=={'model','optimizer','generator','rng','next_phase','Yneg','Y_used','gate',
                    'used_dist','used_gate','phase_seconds','train_seconds'},'Incomplete phase checkpoint state')
    for key in ('Yneg','Y_used'):
        v=st[key];need(isinstance(v,np.ndarray)and v.dtype==np.float32 and v.shape==(n,16)
                      and np.isfinite(v).all()and np.all(v>=0)and np.all(v<=1),'Invalid float32 anchored targets')
    need(torch.is_tensor(st['generator'])and st['generator'].dtype==torch.uint8 and
         set(st['rng'])=={'python','numpy','torch','cuda'},'Missing order/globalRNG state')
    model=st['model']
    need(model['dw.weight'].shape==(16,1,62,1)and model['fc.weight'].shape==(16,256),
         'Wrong EEGNet/classifier shape')
    need(all(torch.is_tensor(v)and torch.isfinite(v).all().item()for v in model.values()),'Nonfinite model/BN checkpoint')
    batches=(n+15)//16
    if n%16==1 and n>16:batches-=1
    calls=st['next_phase']*epochs*batches
    steps=[int(v['step'].item())if torch.is_tensor(v['step'])else int(v['step'])
           for v in st['optimizer']['state'].values()]
    need(len(steps)==12 and set(steps)=={calls},'Adam state/counter mismatch')
    for key in ('bn1.num_batches_tracked','bn2.num_batches_tracked','bn3.num_batches_tracked'):
        need(int(model[key])==1+calls,'BN counter must include original dummy constructor pass plus training')
    need(cp['complete']==(st['next_phase']==phases),'Terminal checkpoint completion mismatch')
    if require_complete:need(cp['complete']is True,'Terminal full-budget checkpoint required')
    return cp


def _fit(X,y,identity,device,checkpoint_path,resume,progress,*,engineering,phases,epochs):
    c=validate_identity(identity,engineering)
    x=np.asarray(X);labels=np.asarray(y)
    need(x.dtype==np.float32 and x.ndim==3 and x.shape[1:]==(62,512)and np.isfinite(x).all(),
         'Only fixed finite normalized float32 Nx62x512 training input')
    n=len(x)
    need((n==100 if not engineering else n in(4,6,8,10,12))and labels.dtype==np.int64
         and labels.shape==(n,)and np.isin(labels,[0,1]).all()
         and np.bincount(labels,minlength=2).tolist()==[n//2,n//2],'Fixed balanced Day1 training labels')
    need((phases==10 and epochs==30)if not engineering else(1<=phases<=4 and 1<=epochs<=2),
         'Only private short QA can override the terminal300 budget')
    dev=torch.device(device);need(dev.type in('cpu','cuda')and(engineering or dev.type=='cuda'),
                                 'No silent campaign CPU fallback')
    if dev.type=='cuda':need(torch.cuda.is_available(),'CUDA requested but unavailable')
    core.set_seed(c['seed'])
    model=core.build_model('EEGNet',62,512,16,PARAMS['dropout']).to(dev)
    initial=model_hash(model)
    immutable=dict(caller=c,input_shape=list(x.shape),input_sha256=tree_hash(x),labels_sha256=tree_hash(labels),
                   phases=phases,epochs_per_phase=epochs,parameters=PARAMS,
                   initialization='Exact original constructor and draw order; no extra initial constrain')
    path=Path(checkpoint_path);saved=None
    if resume:
        cp=load_checkpoint(path,expected_identity=c,allow_engineering=engineering)
        need(cp['identity']==immutable,'Checkpoint normalized data/budget changed')
        saved=cp['state']
    else:need(not path.exists(),'Existing checkpoint requires explicit same-identity resume')
    last={}
    def checkpoint(state):
        payload=dict(schema=SCHEMA,engineering=engineering,identity=immutable,
            identity_canonical=canonical(immutable),state=cpu_copy(state),
            complete=state['next_phase']==phases)
        payload['state_sha256']=tree_hash(payload['state'])
        # Completed audited resume does not rewrite existing source checkpoint.
        _atomic_checkpoint(path,payload);last.update(payload)
    rec,arrays=core.fit_training_only(model,x,labels,walsh=True,rule=RULE,C=data.code_rows(),
        lr=PARAMS['lr'],epochs_per_phase=epochs,batch=PARAMS['batch'],seed=c['seed'],dev=dev,
        phases=phases,resume=saved,checkpoint=checkpoint,progress=progress)
    cp=load_checkpoint(path,expected_identity=c,allow_engineering=engineering,require_complete=True)
    need(model_hash(model)==tree_hash(cp['state']['model'])and tree_hash(x)==immutable['input_sha256']
         and tree_hash(labels)==immutable['labels_sha256'],'Final model/input checkpoint witness')
    record=dict(rec,**c,identity=immutable,initial_model_sha256=initial,
        final_model_sha256=model_hash(model),final_optimizer_sha256=tree_hash(cp['state']['optimizer']),
        final_RNG_sha256=tree_hash(cp['state']['rng']),final_generator_sha256=tree_hash(cp['state']['generator']),
        final_target_sha256=tree_hash(cp['state']['Yneg']),last_used_target_sha256=tree_hash(cp['state']['Y_used']),
        next_phase=phases,campaign_eligible=not engineering,engineering_only=engineering,
        native_training_gate_precision='Unchanged original float32 nearest; W4 itself has no gate',
        native_metric_decoder_precision='Original walsh_metrics casts P tofloat64 before nearest',
        checkpoint_sha256=sha(path),checkpoint_schema=SCHEMA,
        actual_update_calls=phases*epochs*((n+15)//16-(1 if n%16==1 and n>16 else 0)),
        no_test_arrays_accepted=True,terminal_model=True,selection_or_temperature=False,
        training_time_scope='Original elapsed phase loop including checkpoint/progress overhead; no controlled speed comparison')
    outputs=dict(P_train_float32=arrays['P_train'],y_train_int64=arrays['y_train'],
                 Y_used_last_float32=arrays['Y_used_last'])
    return model,record,outputs


def fit(X,y,identity,*,device='cuda',checkpoint_path,resume=False,progress=None):
    return _fit(X,y,identity,device,checkpoint_path,resume,progress,engineering=False,phases=10,epochs=30)


def _fit_synthetic_qa(X,y,identity,*,device='cpu',checkpoint_path,resume=False,progress=None,phases=3,epochs=1):
    return _fit(X,y,identity,device,checkpoint_path,resume,progress,engineering=True,phases=phases,epochs=epochs)
