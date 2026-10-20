#!/usr/bin/env python3
"""Fresh fixed W4 run paired to a completed, same-input public Lee54 work.

The original W4 numerical kernels and custom phase checkpoints are unchanged.
Only local paths, source metadata and public protocol validation are adapted.
No EEG download, historic private workspace or test-dependent selection exists.
"""
from pathlib import Path
import argparse,copy,importlib.util,json,os,shutil,sys
import train as base
HERE=Path(__file__).resolve().parent
TEMPLATE=HERE/'studies/lee_w4/scientific_protocol.public.json'
SOURCE=HERE/'studies/lee_w4/source'
def require(v,m):base.require(v,m)
def read(p):return base.read(p)
def sha(p):return base.sha(p)
def canonical(v):return base.canonical(v)
def expected():return {f'subject_{s:02d}_train__{c}__seed{k}'for s in range(1,55)for c in ('W1','W2')for k in range(1,6)}
def runtime():
 import torch,numpy as np
 return dict(python=sys.version.split()[0],numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)if torch.cuda.is_available()else None,threads=2)

def reference(work):
 pp=work/'PUBLIC_PROTOCOL.json';ps=sha(pp);p,r=base.validate(pp,ps)
 require(p['study']=='lee54','Reference must be a completed public Lee54 experiment, not calibration or a literature score')
 f=read(r/'MODEL_FREEZE.json');t=read(r/'TEST_COMPLETION.json')
 require(f['status']=='ALL1350_FITS_AND54CLASSICAL_FROZEN_BEFORE_DAY2'and f['protocol_sha256']==ps and f['count']==1350 and f['classical']==54 and len(f['files'])==4158,'Complete1350+54 reference model freeze required')
 require(t['status']=='ALL1350_TESTS_AND54CLASSICAL_COMPLETE'and t['protocol_sha256']==ps and t['count']==1350 and t['classical']==54 and len(t['files'])==2808 and t['model_freeze_sha256']==sha(r/'MODEL_FREEZE.json'),'Complete1350+54 reference test gate required')
 for n,h in {**f['files'],**t['files']}.items():
  q=(r/n).resolve();require(q.is_relative_to(r)and sha(q)==h,'Reference fitted/test artifact changed')
 return p,r,ps

def init(reference_work,work):
 ref,r,rps=reference(reference_work.resolve());w=work.resolve();require(not w.exists(),'Use a new W4 work directory')
 dr=Path(ref['data_root']).resolve();require(not dr.is_relative_to(w)and not r.is_relative_to(w),'Input/reference and output roles must be separate')
 template=read(TEMPLATE);w.mkdir(parents=True);code={}
 for rel,h in template['source_code_sha256'].items():
  f=SOURCE/rel;require(sha(f)==h,'Released W4 source changed');q=w/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q);code[rel]=h
 subjects={};readonly={str(r/'PUBLIC_PROTOCOL.json'):rps,str(r/'MODEL_FREEZE.json'):sha(r/'MODEL_FREEZE.json'),str(r/'TEST_COMPLETION.json'):sha(r/'TEST_COMPLETION.json')}
 for s in range(1,55):
  subjects[str(s)]={}
  for role,tail in [('train','train.npz'),('test','test.npz'),('meta','meta.json')]:
   n=f'subject_{s:02d}_{tail}';q=dr/n;h=ref['prepared_inputs'][n];require(sha(q)==h,'Same prepared reference inputs required');subjects[str(s)][role]={'path':str(q),'sha256':h};readonly[str(q)]=h
 historical={}
 for fid in sorted(expected()):
  rp=r/'records'/f'{fid}.json';rec=read(rp);ident=rec['identity'];s=ident['subject'];c=ident['condition'];k=ident['seed']
  require(fid==f'subject_{s:02d}_train__{c}__seed{k}'and rec['n_train']==rec['n_test']==100 and rec['epochs_completed']==300 and rec['phase_test_metrics_computed']is False and rec['final_test_evaluations']==1,'Reference W1/W2 identity/budget')
  for role,tail in [('train','train.npz'),('test','test.npz'),('meta','meta.json')]:require(ident['data_sha256'][tail]==subjects[str(s)][role]['sha256'],'Exact prepared-input pairing required, not matching shapes only')
  pred=(r/rec['prediction_file']).resolve();require(pred.is_relative_to(r)and sha(pred)==rec['prediction_sha256'],'Historical paired prediction hash')
  historical[fid]={'record':{'path':str(rp),'sha256':sha(rp)},'prediction':{'path':str(pred),'sha256':sha(pred)}};readonly[str(rp)]=sha(rp);readonly[str(pred)]=sha(pred)
 b={'status':'BOUND_PUBLIC_REFERENCE_WORK_BEFORE_W4','source_study':str(r),'source_protocol_sha256':rps,'subjects':subjects,'historical':historical,'readonly_sha256':readonly,'protected_files':{},'public_scope':'No private IEEE files; all original reference artifacts checked at initialization; paired record/prediction and prepared bytes prospectively bound.'}
 bp=w/'data/SOURCE_BINDINGS-V1.0.json';base.atomic(bp,b)
 p={**copy.deepcopy(template['protocol']),'public_schema':'portable-w4-with-complete-lee54-reference-v1','created_utc':base.utc(),'controller_sha256':sha(__file__),'base_controller_sha256':sha(HERE/'train.py'),'runtime_versions':base.versions(),'runtime':runtime(),'code_sha256':code,'qa_sha256':{},'source_bindings_sha256':sha(bp),'reference_public_protocol_sha256':rps,'historical_root_protocol_sha256':template['source_protocol_sha256'],'scientific_template_sha256':sha(TEMPLATE),'new_run_not_historical_result':True}
 pp=w/'PUBLIC_W4_PROTOCOL.json';base.atomic(pp,p);print(json.dumps({'protocol':str(pp),'sha256':sha(pp),'reference_W1_W2_records':540,'arrays_decoded':0,'CUDA_required_for_fit':True}))

def validate(path,ps,*,all_sources=True,check_runtime=True):
 pp=Path(path).resolve();w=pp.parent;require(pp.name=='PUBLIC_W4_PROTOCOL.json'and sha(pp)==ps,'W4 public protocol path/hash')
 p=read(pp);require(p['public_schema']=='portable-w4-with-complete-lee54-reference-v1'and p['status']=='FROZEN_BEFORE_ANCHOR_GATE_ABLATION'and p['counts']==dict(fits=270,preprocessors=54,tests=270),'Fixed public W4 scope')
 require(p['controller_sha256']==sha(__file__)and p['base_controller_sha256']==sha(HERE/'train.py'),'Public execution controller changed')
 require(p['scientific_template_sha256']==sha(TEMPLATE),'Released W4 template changed')
 for k,v in read(TEMPLATE)['protocol'].items():require(p[k]==v,'Fixed W4 scientific definition changed '+k)
 require(p['runtime_versions']==base.versions(),'Software changed; use a new W4 work')
 if check_runtime:require(p['runtime']==runtime(),'Frozen W4 hardware/software runtime changed')
 for rel,h in p['code_sha256'].items():
  q=(w/rel).resolve();require(q.is_relative_to(w)and sha(q)==h,'Frozen W4 source changed')
 bp=w/'data/SOURCE_BINDINGS-V1.0.json';require(sha(bp)==p['source_bindings_sha256'],'Public W4 binding changed');b=read(bp)
 require(set(b['subjects'])=={str(s)for s in range(1,55)}and set(b['historical'])==expected()and b['protected_files']=={},'Exact54/540 public binding scope')
 if all_sources:
  for q,h in b['readonly_sha256'].items():require(sha(q)==h,'Reference/prepared source changed '+q)
 return p,w,b

def setup(p,w,b,ps):
 sys.path.insert(0,str(w/'code'));m=base.load('campaign',w/'code/campaign.py');m.ROOT=w;m.PROJECT=w.parent;m.PROTOCOL=w/'PUBLIC_W4_PROTOCOL.json';m.MODEL_FREEZE=w/'campaign/MODEL_FREEZE-V1.0.json';m.TEST_COMPLETION=w/'campaign/TEST_COMPLETION-V1.0.json'
 # This public boundary substitutes complete public reference-work bindings for
 # the historical workstation's12 private submission files. No math is replaced.
 def public_bindings(native):
  require(native['source_bindings_sha256']==sha(w/'data/SOURCE_BINDINGS-V1.0.json'),'Source binding changed')
  return read(w/'data/SOURCE_BINDINGS-V1.0.json')
 m.read_bindings=public_bindings;m._BINDINGS=b;m.install_guard(b);m._PROTOCOL_OK=ps
 import numpy as np
 trains={os.path.normcase(str(Path(v['train']['path']).resolve()))for v in b['subjects'].values()};tests={os.path.normcase(str(Path(v['test']['path']).resolve()))for v in b['subjects'].values()};hist={os.path.normcase(str(Path(v['prediction']['path']).resolve()))for v in b['historical'].values()};sources=trains|tests|hist;own=os.path.normcase(str(w.resolve()))+os.sep
 def audit(event,args):
  if event!='open'or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  q=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
  if Path(q).suffix.lower()not in ('.npz','.npy','.mat','.gdf','.edf','.pt','.ckpt'):return
  if q.startswith(own):return
  require(q in sources,'Numeric source outside public W4 scope')
  mode=args[1]if len(args)>1 else 'r';flags=args[2]if len(args)>2 else 0
  if (isinstance(mode,str)and any(c in mode for c in'wax+'))or (isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):raise PermissionError('Public reference/prepared source is read only')
 old=np.load
 def guarded(file,*a,**kw):
  q=file if isinstance(file,(str,bytes,os.PathLike))else getattr(file,'name',None)
  if isinstance(q,(str,bytes,os.PathLike)):
   n=os.path.normcase(str(Path(q).resolve()))
   if n in tests:require(m._MODEL_GATE is not None and sha(m.MODEL_FREEZE)==m._MODEL_GATE[1],'All270 W4 models must freeze before Day2 decoding')
   if n in hist:require(m._ANALYSIS_GATE,'All270 W4 tests must pass before comparator arrays are decoded')
  return old(file,*a,**kw)
 np.load=guarded;sys.addaudithook(audit);return m

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);a=sub.add_parser('init');a.add_argument('--reference-work',type=Path,required=True);a.add_argument('--work',type=Path,required=True)
 for name in ('validate','fit','freeze','test','analyze'):
  a=sub.add_parser(name);a.add_argument('--protocol',type=Path,required=True);a.add_argument('--protocol-sha256',required=True)
 args=ap.parse_args()
 if args.command=='init':init(args.reference_work,args.work);return
 p,w,b=validate(args.protocol,args.protocol_sha256)
 if args.command=='validate':print(json.dumps({'status':'PASS_PUBLIC_W4_CODE_SOURCE_BINDINGS','paired_comparators':540,'arrays_decoded':0}));return
 m=setup(p,w,b,args.protocol_sha256)
 with base.lock(w):
  if args.command=='fit':
   require(str(m.core.setup_torch(2))=='cuda','W4 campaign requires CUDA; no silent CPU fallback');m.Campaign(p,args.protocol_sha256).train_all()
  elif args.command=='freeze':m.create_model_freeze(p,args.protocol_sha256)
  elif args.command=='test':m.Campaign(p,args.protocol_sha256).test_all()
  else:
   m.validate_test_manifest(p,args.protocol_sha256)
   a=base.load('public_w4_analysis',w/'analysis/final_analysis.py');a.ROOT=w;a.run(w/'PUBLIC_W4_PROTOCOL.json',args.protocol_sha256)
if __name__=='__main__':main()
