#!/usr/bin/env python3
"""Portable prepared-input execution of the added fixed experiments.

No data acquisition is implied. Supply the documented separated-day NPZ inputs.
Historical kernels are copied exactly; this controller supplies fresh local paths,
runtime identity and prospective gates rather than impersonating old local files.
"""
from pathlib import Path
import argparse,contextlib,copy,csv,datetime,hashlib,importlib.util,importlib.metadata,json,os,platform,shutil,sys,time,uuid
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='2'
HERE=Path(__file__).resolve().parent
STUDIES=('lee54','lee_calibration80_20','shu_2x2')
def require(b,msg):
 if not b:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 if p.exists():
  require(p.read_bytes()==b,'Immutable output differs: '+str(p));return
 t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 for k in range(6):
  try:os.replace(t,p);break
  except PermissionError:
   if k==5:raise
   time.sleep(.1*2**k)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def versions():
 return {'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','scipy','torch','scikit-learn','mne','joblib')}}
def init(study,data,work,device):
 data=data.resolve();work=work.resolve();require(data.is_dir(),'Prepared input directory missing')
 require(study!='lee_calibration80_20' or device=='cuda','The byte-preserved historical calibration runner requires CUDA for its memory telemetry; choose --device cuda')
 require(not work.exists(),'Use a new work directory');require(not data.is_relative_to(work),'Inputs must be outside work for phase gating')
 src=HERE/'studies'/study/'source';work.mkdir(parents=True);(work/'code').mkdir();(work/'analysis').mkdir()
 code={}
 for f in src.iterdir():
  if f.name=='final_analysis.py' or f.name=='final_calibration_analysis.py':q=work/'analysis'/f.name
  else:q=work/'code'/f.name
  shutil.copyfile(f,q)
  code[q.relative_to(work).as_posix()]=sha(q)
 protocol=read(HERE/'studies'/study/'scientific_protocol.public.json')['protocol']
 p={'schema':'portable-fixed-prepared-input-study-v1','status':'FROZEN_BEFORE_FRESH_TRAINING','study':study,'created_utc':utc(),
    'device':device,'threads':2,'runtime_versions':versions(),'data_root':str(data),'code_sha256':code,'controller_sha256':sha(__file__),
    'historical_protocol_sha256':read(HERE/'studies'/study/'scientific_protocol.public.json')['source_protocol_sha256'],
    'scientific_protocol':protocol,'prepared_inputs':{},'test_input_paths':[],'new_run_not_historical_result':True}
 if study in ('lee54','lee_calibration80_20'):
  names=[f'subject_{s:02d}_{tail}' for s in range(1,55) for tail in ('train.npz','test.npz','meta.json')]
  for n in names:p['prepared_inputs'][n]=sha(data/n)
  p['test_input_paths']=[str(data/f'subject_{s:02d}_test.npz') for s in range(1,55)]
 else:
  layout=read(HERE/'studies/shu_2x2/input_layout.public.json');b={'day1':{},'later':{},'events_zip_sha256':layout['events_zip_sha256']}
  for s,v in layout['day1'].items():
   b['day1'][s]={}
   for role,z in v.items():
    f=data/z['path'];b['day1'][s][role]={'path':str(f),'sha256':sha(f)};p['prepared_inputs'][z['path']]=sha(f)
  for k,z in layout['later'].items():
   f=data/z['path'];meta=data/z['metadata_path'];b['later'][k]={'path':str(f),'sha256':sha(f),'metadata_path':str(meta),'metadata_sha256':sha(meta)}
   p['prepared_inputs'][z['path']]=sha(f);p['prepared_inputs'][z['metadata_path']]=sha(meta);p['test_input_paths'].append(str(f))
  atomic(work/'data/INPUT_BINDINGS-V1.0.json',b)
  shutil.copyfile(HERE/'studies/shu_2x2/splits.public.json',work/'data/LABELS_AND_INDICES-V1.0.json')
  # Published event metadata is independent of signal values and is pinned separately.
  event_source=HERE/'studies/shu_2x2/events.zip'
  require(sha(event_source)==layout['events_zip_sha256'],'Published event metadata changed')
  (work/'sources').mkdir();shutil.copyfile(event_source,work/'sources/events.zip')
  p['local_bindings_sha256']={f'data/{n}':sha(work/'data'/n) for n in ('INPUT_BINDINGS-V1.0.json','LABELS_AND_INDICES-V1.0.json')}
  p['local_bindings_sha256']['sources/events.zip']=sha(work/'sources/events.zip')
 atomic(work/'PUBLIC_PROTOCOL.json',p);print(json.dumps({'protocol':str(work/'PUBLIC_PROTOCOL.json'),'sha256':sha(work/'PUBLIC_PROTOCOL.json'),'input_files':len(p['prepared_inputs']),'arrays_decoded':0}))
def validate(path,ps,*,all_inputs=True):
 path=Path(path).resolve();require(sha(path)==ps,'Protocol SHA mismatch');p=read(path);w=path.parent
 require(p['schema']=='portable-fixed-prepared-input-study-v1' and p['status']=='FROZEN_BEFORE_FRESH_TRAINING' and p['study'] in STUDIES,'Unsupported protocol')
 require(p['controller_sha256']==sha(__file__),'Controller changed')
 require(p['scientific_protocol']==read(HERE/'studies'/p['study']/'scientific_protocol.public.json')['protocol'],'Scientific template differs from released protocol')
 require(p['runtime_versions']==versions(),'Frozen execution software changed; use a new work directory')
 for k,h in p['code_sha256'].items():require(sha(w/k)==h,'Scientific source changed: '+k)
 for k,h in p.get('local_bindings_sha256',{}).items():require(sha(w/k)==h,'Local binding changed')
 data=Path(p['data_root']).resolve();require(not data.is_relative_to(w),'Input/output role collision')
 for k,h in p['prepared_inputs'].items():
  q=(data/k).resolve();require(q.is_relative_to(data),'Input escapes data root')
  if all_inputs:require(sha(q)==h,'Input bytes changed: '+k)
 return p,w
def runtime(p):
 import numpy as np,torch,scipy
 from lee_core import setup_torch
 dev=setup_torch(2);require(str(dev)==p['device'],'Requested device unavailable; no silent fallback')
 return dev,{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(dev),'threads':2,'deterministic_warnings_only':True}
def guard(p,w):
 state={'test':False};test={os.path.normcase(str(Path(v).resolve())) for v in p['test_input_paths']};inputs={os.path.normcase(str((Path(p['data_root'])/k).resolve())) for k in p['prepared_inputs']};own=os.path.normcase(str(w))+os.sep
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  q=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
  if Path(q).suffix.lower() not in ('.npz','.npy','.mat','.gdf','.edf'):return
  if q.startswith(own):return
  # Hashing bytes uses 'rb'; array parsing is separately guarded at np.load below.
  if q not in inputs:raise PermissionError('Undeclared numeric source: '+q)
  mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 else 0
  if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
   raise PermissionError('Prepared source is read only: '+q)
 import numpy as np
 original=np.load
 def np_load(file,*args,**kwargs):
  target=file if isinstance(file,(str,bytes,os.PathLike)) else getattr(file,'name',None)
  if isinstance(target,(str,bytes,os.PathLike)):
   q=os.path.normcase(str(Path(target).resolve()))
   if q in test and not state['test']:raise PermissionError('All fitted artifacts must freeze before test arrays')
  return original(file,*args,**kwargs)
 np.load=np_load;sys.addaudithook(audit);return state
@contextlib.contextmanager
def lock(w):
 q=w/'PUBLIC_RUNNING.lock'
 with q.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'host':platform.node(),'started':utc()},f)
 try:yield
 finally:q.unlink(missing_ok=True)
def modules(w):
 sys.path.insert(0,str(w/'code'))
def cal_command(command,p,w,ps,gate):
 modules(w);m=load('calibration_runner',w/'code/calibration_runner.py');m.STUDY=w;m.HERE=w/'code'
 def verify(q,h,all_inputs=True):
  pp,ww=validate(q,h,all_inputs=all_inputs);require(ww==w and pp==p,'Protocol context mismatch')
  return {'_protocol_sha256':ps,'_path':str(w/'PUBLIC_PROTOCOL.json'),'data':{'prepared_dir':p['data_root'],'prepared_sha256':p['prepared_inputs']},'execution':{'device':p['device'],'threads':2,'parallel_gpu_workers':1}}
 # The native driver hashes the current train/meta files per fit and test file
 # per evaluation. Its per-job immutable-code check does not rehash every EEG
 # file. Full input hashing still occurs at the start of every public command.
 m.verify_protocol=verify;m.verify_current=lambda x:verify(w/'PUBLIC_PROTOCOL.json',ps,all_inputs=False)
 m.runtime_setup=lambda x:runtime(p);cp=verify(w/'PUBLIC_PROTOCOL.json',ps)
 if command=='fit':m.fit_models(cp,recover=False)
 elif command=='freeze':m.freeze_models(cp,recover=False)
 elif command=='test':m.verify_global_manifest(cp);gate['test']=True;m.evaluate_models(cp,recover=False)
 elif command=='analyze':
  m.verify_global_manifest(cp);gate['test']=True
  a=load('public_cal_analysis',w/'analysis/final_calibration_analysis.py');a.ROOT=w
  old=sys.argv;sys.argv=[str(w/'analysis/final_calibration_analysis.py'),'--protocol',str(w/'PUBLIC_PROTOCOL.json'),'--protocol-sha256',ps]
  try:a.main()
  finally:sys.argv=old
def shu_command(command,p,w,ps,gate):
 modules(w);m=load('campaign',w/'code/campaign.py');m.ROOT=w;m.data.ROOT=w;m.data.PROJECT=w.parent
 native=copy.deepcopy(p['scientific_protocol']);native['code_sha256']=p['code_sha256'];native['protected_files']={}
 # Preserve the historical gate's fixed filename while linking the fresh public protocol bytes.
 native['counts']=m.data.COUNTS
 native['data_sha256']=p['local_bindings_sha256'];native['qa_sha256']={};native['old_scientific_protection']={}
 native['events_metadata_path']=str(w/'sources/events.zip');native['readonly_day1_inputs']={}
 native['_audit_file_counts']={'code_sha256':len(p['code_sha256']),'data_sha256':len(p['local_bindings_sha256']),'qa_sha256':0,'protected_files':0,'old_scientific_protection':0,'readonly_day1_inputs':len(p['prepared_inputs'])}
 # Native validation checks this file's bytes. It is an exact public-protocol copy, not an old protocol.
 target=w/'TAM_DENEY_PROTOKOLU-V1.0.json'
 if not target.exists():shutil.copyfile(w/'PUBLIC_PROTOCOL.json',target)
 require(sha(target)==ps,'Native public-protocol bridge mismatch')
 m.data.install_guard();c=m.Campaign(native,ps,w/'PUBLIC_PROTOCOL.json',resume=True)
 if command=='fit':c.cv();c.main_fits()
 elif command=='freeze':c.freeze()
 elif command=='test':
  fp=w/'campaign/MODEL_FREEZE-V1.0.json';m.data.validate_freeze(native,ps,fp);gate['test']=True;c.tests(fp)
 elif command=='analyze':
  # The original analysis independently rechecks every fit/test/hash. Its path/runtime
  # preflight is replaced only by the public role-bound protocol validation above.
  fp=w/'campaign/MODEL_FREEZE-V1.0.json';m.data.validate_freeze(native,ps,fp);gate['test']=True
  a=load('public_shu_analysis',w/'analysis/final_analysis.py');a.ROOT=w
  def analysis_protocol(path,digest,root):
   pp,ww=validate(path,digest);require(ww==w and pp==p,'Analysis protocol context mismatch');return copy.deepcopy(native)
  a.verify_protocol=analysis_protocol
  old=sys.argv;sys.argv=[str(w/'analysis/final_analysis.py'),'--protocol',str(w/'PUBLIC_PROTOCOL.json'),'--protocol-sha256',ps]
  try:a.main()
  finally:sys.argv=old
def lee_jobs():return [(s,c,k) for s in range(1,55) for c in ('S0','W0','W1','W2','W3') for k in range(1,6)]
def lee_key(s,c,k):return f'subject_{s:02d}_train__{c}__seed{k}'
def npz_atomic(p,**a):
 import numpy as np
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.stem+'.tmp-'+uuid.uuid4().hex+'.npz');np.savez_compressed(tmp,**a);os.replace(tmp,p)
def lee_context(p,w,ps):
 modules(w)
 import lee_core as core,lee_data as data
 dev,env=runtime(p)
 identity={'command':'run','dataset':'Lee2019_MI','code_sha256':p['code_sha256'],'public_controller_sha256':p['controller_sha256'],'public_protocol_sha256':ps,'parameters':data.PARAMS,**env}
 return core,data,dev,identity
def lee_fit(p,w,ps):
 import numpy as np,torch
 core,data,dev,common=lee_context(p,w,ps);dr=Path(p['data_root'])
 for s,c,k in lee_jobs():
  key=lee_key(s,c,k);rp=w/'training_records'/f'{key}.json';fp=w/'fit_arrays'/f'{key}.npz';cp=w/'checkpoints'/f'{key}.pt'
  ident={**common,'subject':s,'condition':c,'seed':k,'data_sha256':{t:p['prepared_inputs'][f'subject_{s:02d}_{t}'] for t in ('train.npz','test.npz','meta.json')}}
  if rp.exists():
   r=read(rp);require(r['identity']==ident and r['checkpoint_sha256']==sha(cp) and r['fit_arrays_sha256']==sha(fp),'Completed fit identity/artifact changed');continue
  meta=read(dr/f'subject_{s:02d}_meta.json');require(meta.get('subject')==s and meta.get('data_units',meta.get('units'))=='uV','Lee subject/unit metadata')
  raw,y=data.load_part(dr/f'subject_{s:02d}_train.npz','train');mean,sd=data.zscore_fit(raw);x=data.zscore_apply(raw,mean,sd)
  require(raw.shape==(100,62,512),'Prepared Lee contract is100x62x512 at128Hz')
  core.set_seed(k);model=core.build_model('EEGNet',62,512,2 if c=='S0' else 16,.5).to(dev)
  partial=w/'partial_checkpoints'/f'{key}.pt';resume=None
  if partial.exists():
   saved=torch.load(partial,map_location=dev,weights_only=False);require(saved['identity']==ident,'Phase checkpoint identity changed');resume=saved['state']
   resume['generator']=resume['generator'].cpu();resume['rng']['torch']=resume['rng']['torch'].cpu()
   if resume['rng']['cuda'] is not None:resume['rng']['cuda']=[z.cpu() for z in resume['rng']['cuda']]
  def checkpoint(state):
   partial.parent.mkdir(parents=True,exist_ok=True);tmp=partial.with_suffix('.tmp.pt');torch.save({'identity':ident,'state':state},tmp);os.replace(tmp,partial)
  rec,arr=core.fit_training_only(model,x,y,walsh=c!='S0',rule=data.RULES[c],C=data.code_rows(),lr=.001,epochs_per_phase=30,batch=16,seed=k,dev=dev,phases=10,resume=resume,checkpoint=checkpoint)
  cp.parent.mkdir(parents=True,exist_ok=True);tmp=cp.with_suffix('.tmp.pt');torch.save({'identity':ident,'model_state':model.state_dict()},tmp);os.replace(tmp,cp)
  npz_atomic(fp,**arr,normalization_mean=mean,normalization_sd=sd)
  atomic(rp,{'id':key,'identity':ident,'train_record':rec,'checkpoint_path':cp.relative_to(w).as_posix(),'checkpoint_sha256':sha(cp),'fit_arrays_path':fp.relative_to(w).as_posix(),'fit_arrays_sha256':sha(fp),'n_train':100,'fit_complete':True,'test_arrays_opened':False})
  print(json.dumps({'fit_complete':key}),flush=True)
 # The contextual reference uses its original two-ended CSP and default SVD LDA.
 # Fit all54 before the same global Day2 gate; no test array enters this path.
 import classical_reference as classical,joblib
 from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 for s in range(1,55):
  rp=w/'classical_fit_records'/f'subject_{s:02d}.json';hp=w/'classical_heads'/f'subject_{s:02d}.joblib'
  if rp.exists():
   r=read(rp);require(r['protocol_sha256']==ps and sha(hp)==r['head_sha256'],'Classical fit identity changed');continue
  x,y=data.load_part(dr/f'subject_{s:02d}_train.npz','train');W,ix=classical.fit_csp(x,y);features=classical.csp_transform(x,W)
  head=LinearDiscriminantAnalysis().fit(features,y);require(head.classes_.tolist()==[0,1],'Classical class order')
  hp.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'W':W,'selected_indices':ix,'lda':head},hp)
  atomic(rp,{'subject':s,'protocol_sha256':ps,'head_path':hp.relative_to(w).as_posix(),'head_sha256':sha(hp),'fit_trials':100,'features':6,'test_arrays_opened':False})
def lee_freeze(p,w,ps):
 ids={lee_key(*j) for j in lee_jobs()};require({f.stem for f in (w/'training_records').glob('*.json')}==ids,'All1350 fits required')
 files={}
 for fid in sorted(ids):
  rp=w/'training_records'/f'{fid}.json';r=read(rp);require(r['identity']['public_protocol_sha256']==ps and r['fit_complete'] and not r['test_arrays_opened'],'Fit identity/test boundary')
  require(r['train_record']['epochs_completed']==300 and r['n_train']==100,'Training budget changed')
  for path,h in [(r['checkpoint_path'],r['checkpoint_sha256']),(r['fit_arrays_path'],r['fit_arrays_sha256'])]:
   q=(w/path).resolve();require(q.is_relative_to(w) and sha(q)==h,'Fit artifact mismatch');files[path]=h
  files[rp.relative_to(w).as_posix()]=sha(rp)
 require({f.stem for f in (w/'classical_fit_records').glob('*.json')}=={f'subject_{s:02d}' for s in range(1,55)},'All54classical fits required')
 for rp in sorted((w/'classical_fit_records').glob('*.json')):
  r=read(rp);require(r['protocol_sha256']==ps and r['fit_trials']==100 and r['features']==6 and not r['test_arrays_opened'],'Classical budget/boundary')
  require(sha(w/r['head_path'])==r['head_sha256'],'Classical fit artifact changed');files[r['head_path']]=r['head_sha256'];files[rp.relative_to(w).as_posix()]=sha(rp)
 atomic(w/'MODEL_FREEZE.json',{'status':'ALL1350_FITS_AND54CLASSICAL_FROZEN_BEFORE_DAY2','protocol_sha256':ps,'files':files,'count':1350,'classical':54})
def lee_test(p,w,ps,gate):
 import numpy as np,torch
 core,data,dev,common=lee_context(p,w,ps);freeze=read(w/'MODEL_FREEZE.json');require(freeze['protocol_sha256']==ps and freeze['count']==1350,'Incomplete model freeze')
 for f,h in freeze['files'].items():require(sha(w/f)==h,'Frozen fit artifact changed')
 require(len(freeze['files'])==4158 and freeze['classical']==54,'Wrong all-fit union');gate['test']=True
 for s,c,k in lee_jobs():
  key=lee_key(s,c,k);rp=w/'records'/f'{key}.json';pp=w/'predictions'/f'{key}.npz';fit=read(w/'training_records'/f'{key}.json')
  if rp.exists():
   r=read(rp);require(r['identity']==fit['identity'] and sha(pp)==r['prediction_sha256'],'Existing test changed');continue
  cp=torch.load(w/fit['checkpoint_path'],map_location=dev,weights_only=False);require(cp['identity']==fit['identity'],'Checkpoint identity')
  core.set_seed(k);m=core.build_model('EEGNet',62,512,2 if c=='S0' else 16,.5).to(dev);m.load_state_dict(cp['model_state'],strict=True)
  with np.load(w/fit['fit_arrays_path'],allow_pickle=False) as a:arr={n:a[n] for n in a.files}
  raw,y=data.load_part(Path(p['data_root'])/f'subject_{s:02d}_test.npz','test')
  final,ta=core.evaluate_final(m,data.zscore_apply(raw,arr['normalization_mean'],arr['normalization_sd']),y,walsh=c!='S0',C=data.code_rows(),dev=dev)
  npz_atomic(pp,**arr,**ta);r={**fit['train_record'],**final,'identity':fit['identity'],'job':key,'n_train':100,'n_test':100,'prediction_file':pp.relative_to(w).as_posix(),'prediction_sha256':sha(pp),'all_model_freeze_sha256':sha(w/'MODEL_FREEZE.json'),'completed_utc':utc()};atomic(rp,r)
 import classical_reference as classical,joblib
 for s in range(1,55):
  rp=w/'classical_results'/f'subject_{s:02d}.json';pp=w/'classical_predictions'/f'subject_{s:02d}.npz'
  if rp.exists():
   r=read(rp);require(r['protocol_sha256']==ps and sha(pp)==r['prediction_sha256'],'Classical test changed');continue
  fit=read(w/'classical_fit_records'/f'subject_{s:02d}.json');head=joblib.load(w/fit['head_path'])
  x,y=data.load_part(Path(p['data_root'])/f'subject_{s:02d}_test.npz','test');prob=head['lda'].predict_proba(classical.csp_transform(x,head['W']))
  npz_atomic(pp,P_test=prob,y_test=y)
  atomic(rp,{'subject':s,'protocol_sha256':ps,'metrics':classical.class_metrics(prob,y),'prediction_sha256':sha(pp),'role':'separate descriptive six-feature both-ended classical reference'})
 ids={lee_key(*j) for j in lee_jobs()};require({f.stem for f in (w/'records').glob('*.json')}==ids,'Incomplete test records')
 require({f.stem for f in (w/'classical_results').glob('*.json')}=={f'subject_{s:02d}' for s in range(1,55)},'All54classical tests required')
 atomic(w/'TEST_COMPLETION.json',{'status':'ALL1350_TESTS_AND54CLASSICAL_COMPLETE','protocol_sha256':ps,'model_freeze_sha256':sha(w/'MODEL_FREEZE.json'),'count':1350,'classical':54,'files':{f.relative_to(w).as_posix():sha(f) for sub in ('records','predictions','classical_results','classical_predictions') for f in (w/sub).iterdir()}})
def lee_analyze(p,w,ps,gate):
 comp=read(w/'TEST_COMPLETION.json');require(comp['protocol_sha256']==ps and comp['count']==1350,'Complete tests required')
 for n,h in comp['files'].items():require(sha(w/n)==h,'Test artifact changed')
 require(comp['classical']==54 and len(comp['files'])==2808,'Complete classical/neural test union required')
 csvpath=w/'classical_summary.csv'
 if not csvpath.exists():
  with csvpath.open('x',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,['subject','acc_test']);writer.writeheader()
   for s in range(1,55):writer.writerow({'subject':s,'acc_test':read(w/'classical_results'/f'subject_{s:02d}.json')['metrics']['acc']})
 gate['test']=True;a=load('public_lee_analysis',w/'analysis/final_analysis.py');old=sys.argv;sys.argv=['analysis','--results',str(w),'--out',str(w/'final_analysis'),'--csp-csv',str(csvpath)]
 try:a.main()
 finally:sys.argv=old
def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
 a=sub.add_parser('init');a.add_argument('--study',choices=STUDIES,required=True);a.add_argument('--data-root',type=Path,required=True);a.add_argument('--work',type=Path,required=True);a.add_argument('--device',choices=('cpu','cuda'),default='cuda')
 for name in ('validate','fit','freeze','test','analyze'):
  a=sub.add_parser(name);a.add_argument('--protocol',type=Path,required=True);a.add_argument('--protocol-sha256',required=True)
 a=ap.parse_args()
 if a.command=='init':init(a.study,a.data_root,a.work,a.device);return
 p,w=validate(a.protocol,a.protocol_sha256)
 if a.command=='validate':print(json.dumps({'status':'PASS_PUBLIC_INPUT_CODE_HASHES','arrays_decoded':0,'input_files':len(p['prepared_inputs'])}));return
 if p['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
 gate=guard(p,w)
 with lock(w):
  if p['study']=='lee54':
   {'fit':lee_fit,'freeze':lee_freeze,'test':lambda p,w,ps:lee_test(p,w,ps,gate),'analyze':lambda p,w,ps:lee_analyze(p,w,ps,gate)}[a.command](p,w,a.protocol_sha256)
  elif p['study']=='lee_calibration80_20':cal_command(a.command,p,w,a.protocol_sha256,gate)
  else:shu_command(a.command,p,w,a.protocol_sha256,gate)
if __name__=='__main__':main()
