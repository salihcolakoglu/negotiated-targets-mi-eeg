"""Fit-only EEGNet/FBCNet x CE/Walsh engine with exact resumable state.

The engineering entry is explicitly ineligible for a future full campaign.
This module never loads EEG files, computes outcome metrics, selects models or
fits calibration.  It accepts only balanced50target or630/700source fit arrays.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import struct
import time
import uuid
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F
import models

ARCHITECTURES=('EEGNet','FBCNet')
OUTPUTS=('CE','WALSH')
BATCH_SIZE=16
LEARNING_RATE=.001
CHECKPOINT_MAGIC=b'SHU_FBCNET_CE_WALSH_CHECKPOINT_V1\x00'
CHECKPOINT_SCHEMA='shu-fbcnet-ce-walsh-fit-block-v1'


def _frame(d,b):d.update(len(b).to_bytes(8,'big'));d.update(b)


def _tensor_bytes(t):return t.detach().cpu().contiguous().numpy().tobytes(order='C')


def state_dict_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        _frame(h,json.dumps(dict(name=name,dtype=str(t.dtype),shape=list(t.shape)),sort_keys=True,separators=(',',':')).encode('ascii'))
        _frame(h,_tensor_bytes(t))
    return h.hexdigest()


def state_hash(model):return state_dict_hash(model.state_dict())


def backbone_hash(model):
    # Both architectures must expose the output head as fc. The head is the
    # only state excluded from this shared-initial-backbone audit fingerprint.
    if not isinstance(getattr(model,'fc',None),torch.nn.Module):
        raise TypeError('Models must expose the output module as fc for the backbone pairing audit')
    return state_dict_hash({k:v for k,v in model.state_dict().items() if not k.startswith('fc.')})


def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    for b in (str(a.dtype).encode('ascii'),json.dumps(list(a.shape)).encode('ascii'),a.tobytes()):_frame(h,b)
    return h.hexdigest()


def _generator_hash(g):return hashlib.sha256(_tensor_bytes(g.get_state())).hexdigest()


def _capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _rng_hash(state):
    h=hashlib.sha256();_frame(h,repr(state['python']).encode('ascii'))
    ns=state['numpy'];_frame(h,repr((ns[0],ns[2],ns[3],ns[4])).encode('ascii'));_frame(h,np.ascontiguousarray(ns[1]).tobytes())
    _frame(h,_tensor_bytes(state['torch_cpu']))
    for v in state['torch_cuda'] or []:_frame(h,_tensor_bytes(v))
    return h.hexdigest()


def _restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch_cpu'].cpu())
    if r['torch_cuda'] is not None:
        if not torch.cuda.is_available():raise RuntimeError('CUDA RNG checkpoint cannot resume without CUDA')
        torch.cuda.set_rng_state_all([v.cpu() for v in r['torch_cuda']])


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def _finite_model(model):
    for name,v in model.state_dict().items():
        if not torch.isfinite(v).all().item():raise FloatingPointError('Nonfinite model parameter/buffer: '+name)


def _strict_initial(model,state):
    if not isinstance(state,Mapping):raise TypeError('initial_state must contain the complete same-architecture/output source state')
    expected=model.state_dict()
    if set(expected)!=set(state):raise ValueError('Source state keys differ from complete target network')
    for name,v in state.items():
        if not torch.is_tensor(v) or v.dtype!=expected[name].dtype or v.shape!=expected[name].shape:
            raise ValueError('Source/target architecture or output shape/dtype mismatch: '+name)
        if not torch.isfinite(v).all().item():raise FloatingPointError('Nonfinite source state: '+name)
    model.load_state_dict(copy.deepcopy(state),strict=True)


def _optimizer_steps(opt):
    steps=[]
    for state in opt.state.values():
        v=state.get('step')
        if v is not None:steps.append(int(v.item()) if torch.is_tensor(v) else int(v))
    return steps


def _atomic_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=io.BytesIO();torch.save(payload,b);raw=b.getvalue()
    envelope=CHECKPOINT_MAGIC+struct.pack('>Q',len(raw))+hashlib.sha256(raw).digest()+raw
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:f.write(envelope);f.flush();os.fsync(f.fileno())
        for attempt in range(6):
            try:os.replace(tmp,path);break
            except OSError as exc:
                if getattr(exc,'winerror',None) not in (5,32,33) and not isinstance(exc,PermissionError):raise
                if attempt==5:raise RuntimeError('Checkpoint replacement failed after6bounded attempts') from exc
                time.sleep(.1*2**attempt)
    finally:
        if tmp.exists():tmp.unlink()


def load_checkpoint(path):
    """Validate this local study's SHA envelope, then deserialize trusted state.

    SHA guards corruption, not authenticity; never use for arbitrary downloaded
    pickle files. Full Adam and Python/NumPy RNG require trusted deserialization.
    """
    raw=Path(path).read_bytes();offset=len(CHECKPOINT_MAGIC);header=offset+8+32
    if len(raw)<header or raw[:offset]!=CHECKPOINT_MAGIC:raise RuntimeError('Invalid checkpoint envelope header')
    size=struct.unpack('>Q',raw[offset:offset+8])[0];digest=raw[offset+8:header];payload=raw[header:]
    if len(payload)!=size or hashlib.sha256(payload).digest()!=digest:raise RuntimeError('Checkpoint SHA256/size mismatch')
    obj=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    if not isinstance(obj,dict) or obj.get('checkpoint_schema')!=CHECKPOINT_SCHEMA:raise RuntimeError('Checkpoint schema mismatch')
    return obj


def _check_predictions(l,p,n,output):
    width=2 if output=='CE' else 16
    for a in (l,p):
        if not isinstance(a,np.ndarray) or a.dtype!=np.float32 or a.shape!=(n,width) or not np.isfinite(a).all():
            raise FloatingPointError('Invalid fit-only float32 logits/probabilities')
    if np.any(p<0) or np.any(p>1):raise FloatingPointError('Fit-only probabilities outside[0,1]')
    if output=='CE' and not np.allclose(p.sum(1),1.,rtol=0,atol=2e-6):raise FloatingPointError('CE probabilities do not sum to1')


@torch.no_grad()
def _fit_outputs(model,x,architecture,output):
    model.eval();before=state_hash(model)
    logits,probability=models.predict_outputs(model,x)
    if torch.is_tensor(logits):logits=logits.detach().cpu().numpy()
    if torch.is_tensor(probability):probability=probability.detach().cpu().numpy()
    logits=np.asarray(logits);probability=np.asarray(probability)
    _check_predictions(logits,probability,len(x),output)
    chunks=[]
    for start in range(0,len(x),256):
        value=model.features(x[start:start+256])
        chunks.append(value.detach().float().cpu().numpy())
    features=np.concatenate(chunks)
    width=256 if architecture=='EEGNet' else 1152
    if features.dtype!=np.float32 or features.shape!=(len(x),width) or not np.isfinite(features).all():
        raise FloatingPointError('Invalid frozen '+architecture+' feature matrix')
    if state_hash(model)!=before:raise RuntimeError('Prediction/features mutated model state; eval must not constrain/update weights')
    return features,logits,probability


def _train_epochs(model,opt,x,y,targets,output,epochs,generator):
    n=len(x)
    for _ in range(epochs):
        model.train()
        order=torch.randperm(n,generator=generator)
        for start in range(0,n,BATCH_SIZE):
            idx=order[start:start+BATCH_SIZE].to(x.device)
            opt.zero_grad(set_to_none=True)
            logits=model(x[idx])
            width=2 if output=='CE' else 16
            if logits.dtype!=torch.float32 or tuple(logits.shape)!=(len(idx),width) or not torch.isfinite(logits).all().item():
                raise FloatingPointError('Nonfinite/wrong-shaped training logits')
            loss=F.cross_entropy(logits,y[idx]) if output=='CE' else F.binary_cross_entropy_with_logits(logits,targets[idx])
            if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad and (v.grad is None or not torch.isfinite(v.grad).all().item()):
                    raise FloatingPointError('Absent/nonfinite training gradient: '+name)
            opt.step()
            with torch.no_grad():model.constrain()


def fit_engineering(X,y,architecture,output,seed_init,seed_batch,device,*,epochs=30,
                    initial_state=None,checkpoint_path=None,resume=False,progress=None,
                    engineering_label='SOURCE21_PILOT',run_fingerprint=None):
    """Engineering/source-only pilot, default2x15epochs; never campaign eligible.

    Short even budgets2..30 are allowed solely for explicitly named synthetic
    engineering QA. A caller can interrupt via progress(block,blocks,seconds),
    which runs only after the complete block checkpoint has been committed.
    """
    if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<2 or epochs>30 or epochs%2:
        raise ValueError('Engineering budget must be an even integer2..30; real source pilot uses30')
    if not isinstance(engineering_label,str) or not engineering_label.strip():raise ValueError('Engineering label is required')
    return _fit(X,y,architecture,output,seed_init,seed_batch,device,blocks=2,epochs_per_block=epochs//2,
        initial_state=initial_state,checkpoint_path=checkpoint_path,resume=resume,progress=progress,
        engineering_label=engineering_label,run_fingerprint=run_fingerprint)


def fit_main(X,y,architecture,output,seed_init,seed_batch,device,*,initial_state=None,
             checkpoint_path=None,resume=False,progress=None,run_fingerprint=None):
    """Future separately authorized campaign entry: exactly10x30epochs.

    Defining this API does not launch or authorize a full campaign. Current real
    activity is limited to the separate source-only engineering entry above.
    """
    return _fit(X,y,architecture,output,seed_init,seed_batch,device,blocks=10,epochs_per_block=30,
        initial_state=initial_state,checkpoint_path=checkpoint_path,resume=resume,progress=progress,
        engineering_label=None,run_fingerprint=run_fingerprint)


def _fit(X,y,architecture,output,seed_init,seed_batch,device,*,blocks,epochs_per_block,
         initial_state,checkpoint_path,resume,progress,engineering_label,run_fingerprint):
    if architecture not in ARCHITECTURES or output not in OUTPUTS:raise ValueError('Expected EEGNet/FBCNet and CE/WALSH')
    features=np.asarray(X);labels=np.asarray(y)
    shape=(29,512) if architecture=='EEGNet' else(9,29,512)
    if features.dtype!=np.float32 or features.ndim!=len(shape)+1 or features.shape[1:]!=shape or len(features) not in (50,630,700):
        raise ValueError('Expected float32 50/630/700 x'+str(shape)+' fit inputs')
    if not np.isfinite(features).all():raise FloatingPointError('Nonfinite training inputs')
    if labels.dtype!=np.int64 or labels.shape!=(len(features),) or np.bincount(labels,minlength=2).tolist()!=[len(features)//2]*2:
        raise ValueError('Expected exactly balanced binary int64 fit labels')
    if initial_state is not None and len(features)!=50:raise ValueError('Only target50 fits can initialize from source state')
    for seed in (seed_init,seed_batch):
        if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=int(seed)<2**32:raise ValueError('Seed must be integer in[0,2**32)')
    if checkpoint_path is not None:
        checkpoint_path=Path(checkpoint_path)
        if not isinstance(run_fingerprint,dict) or not run_fingerprint:raise ValueError('Checkpoint requires nonempty immutable run_fingerprint')
        if checkpoint_path.exists() and not resume:raise FileExistsError('Checkpoint exists; explicit validated resume required')
        if resume and not checkpoint_path.is_file():raise FileNotFoundError('Resume checkpoint missing')
    elif resume:raise ValueError('Resume requires checkpoint_path')
    dev=torch.device(device);mode='DIRECT' if initial_state is not None else'FIXED'
    started=time.perf_counter();source_before=None if initial_state is None else state_dict_hash(initial_state)
    baseline=0
    if dev.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        baseline=int(torch.cuda.memory_allocated(dev));torch.cuda.reset_peak_memory_stats(dev)
    models.set_seed(int(seed_init));build_start=time.perf_counter()
    model=models.build_model(architecture,output).to(dev)
    if not callable(getattr(model,'constrain',None)):raise TypeError('Model must expose mutation-free-eval and explicit constrain()')
    with torch.no_grad():model.constrain()
    _sync(dev);build_seconds=time.perf_counter()-build_start
    fresh=state_hash(model);fresh_backbone=backbone_hash(model);load_seconds=0.
    if initial_state is not None:
        tick=time.perf_counter();_strict_initial(model,initial_state);_sync(dev);load_seconds=time.perf_counter()-tick
    _finite_model(model);initial=state_hash(model)
    if source_before is not None and initial!=source_before:raise RuntimeError('Complete source state did not transfer exactly')
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    if opt.state:raise RuntimeError('Fresh Adam unexpectedly has moments')
    # Head-width random draws must not change later dropout randomness. This
    # reset pairs stochastic training independently of CE2 versus Walsh16 init.
    models.set_seed(int(seed_batch));initial_rng=_rng_hash(_capture_rng())
    generator=torch.Generator().manual_seed(int(seed_batch));generator_initial=_generator_hash(generator)
    xt=torch.from_numpy(np.ascontiguousarray(features)).to(dev);yt=torch.from_numpy(labels.copy()).to(dev)
    codes=np.asarray([[1]*16,[1,0]*8],dtype=np.float32)
    target_array=labels.copy() if output=='CE' else codes[labels].copy()
    targets=yt if output=='CE' else torch.from_numpy(target_array).to(dev)
    modules=(Path(__file__),Path(models.__file__))
    identity=dict(run_fingerprint=run_fingerprint,module_sha256={m.name:hashlib.sha256(m.read_bytes()).hexdigest() for m in modules},
        architecture=architecture,output=output,mode=mode,features_sha256=array_hash(features),labels_sha256=array_hash(labels),
        targets_sha256=array_hash(target_array),source_initial_state_sha256=source_before,
        initial_fresh_state_sha256=fresh,initial_fresh_backbone_sha256=fresh_backbone,initial_training_state_sha256=initial,
        batch_generator_initial_sha256=generator_initial,training_global_RNG_initial_sha256=initial_rng,
        seed_init=int(seed_init),seed_batch=int(seed_batch),training_rng_seed=int(seed_batch),device=str(dev),
        config=dict(blocks=blocks,epochs_per_block=epochs_per_block,batch_size=BATCH_SIZE,learning_rate=LEARNING_RATE,
            engineering_label=engineering_label,loss='cross_entropy' if output=='CE' else'binary_cross_entropy_with_logits',
            constraints='initial fresh model and after every Adam step; never in eval forward'),
        runtime=dict(torch=str(torch.__version__),numpy=str(np.__version__),cuda=torch.version.cuda,
            GPU=torch.cuda.get_device_name(dev) if dev.type=='cuda' else None,threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark))
    canonical=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
    start_block=0;seconds_before=0.;resume_count=0;block_seconds=[];peak_before=reserved_before=0
    logits=probability=fit_features=None;updates=(len(features)+BATCH_SIZE-1)//BATCH_SIZE
    if resume:
        cp=load_checkpoint(checkpoint_path)
        if cp.get('identity_canonical')!=canonical:raise RuntimeError('Resume identity changed: code/data/source/seeds/output/config/environment')
        start_block=cp['completed_blocks']
        if isinstance(start_block,bool) or not isinstance(start_block,int) or not 1<=start_block<=blocks:raise RuntimeError('Invalid completed checkpoint block')
        _strict_initial(model,cp['model_state'])
        if state_hash(model)!=cp['model_state_sha256']:raise RuntimeError('Checkpoint model state SHA mismatch')
        model.train(bool(cp['model_training']));opt.load_state_dict(cp['optimizer_state'])
        expected_steps=start_block*epochs_per_block*updates;steps=_optimizer_steps(opt)
        if not steps or any(n!=expected_steps for n in steps):raise RuntimeError('Checkpoint Adam steps inconsistent with completed epochs')
        generator.set_state(cp['sample_generator_state'].cpu())
        if _generator_hash(generator)!=cp['sample_generator_sha256']:raise RuntimeError('Checkpoint sample order RNG SHA mismatch')
        if not np.array_equal(cp['last_targets'],target_array):raise RuntimeError('Checkpoint labels/hard targets changed')
        logits=cp['fit_logits'].copy();probability=cp['previous_prediction'].copy();fit_features=cp['fit_features'].copy()
        _check_predictions(logits,probability,len(features),output)
        if fit_features.dtype!=np.float32 or fit_features.shape!=(len(features),256 if architecture=='EEGNet' else 1152) or not np.isfinite(fit_features).all():
            raise RuntimeError('Checkpoint feature outputs invalid')
        block_seconds=list(cp['block_seconds'])
        if len(block_seconds)!=start_block:raise RuntimeError('Checkpoint block history length mismatch')
        seconds_before=float(cp['train_seconds']);resume_count=int(cp['resume_count'])+1
        build_seconds=float(cp['network_construction_seconds']);load_seconds=float(cp['initial_state_load_seconds'])
        peak_before=int(cp['max_cuda_memory_allocated_bytes']);reserved_before=int(cp['max_cuda_memory_reserved_bytes'])
        _finite_model(model)
        if _rng_hash(cp['rng_state'])!=cp['rng_state_sha256']:raise RuntimeError('Checkpoint global RNG hash mismatch')
        _restore_rng(cp['rng_state'])
    _sync(dev);training_started=time.perf_counter()
    for block in range(start_block,blocks):
        _sync(dev);tick=time.perf_counter()
        _train_epochs(model,opt,xt,yt,targets,output,epochs_per_block,generator)
        _finite_model(model);fit_features,logits,probability=_fit_outputs(model,xt,architecture,output)
        _sync(dev);block_seconds.append(float(time.perf_counter()-tick));elapsed=seconds_before+time.perf_counter()-training_started
        if checkpoint_path is not None:
            rng=_capture_rng()
            _atomic_checkpoint(checkpoint_path,dict(checkpoint_schema=CHECKPOINT_SCHEMA,identity_canonical=canonical,
                run_fingerprint=run_fingerprint,engineering_only=engineering_label is not None,completed_blocks=block+1,
                model_state=model.state_dict(),model_training=bool(model.training),model_state_sha256=state_hash(model),
                optimizer_state=opt.state_dict(),sample_generator_state=generator.get_state(),sample_generator_sha256=_generator_hash(generator),
                rng_state=rng,rng_state_sha256=_rng_hash(rng),last_targets=target_array.copy(),previous_prediction=probability.copy(),
                fit_logits=logits.copy(),fit_features=fit_features.copy(),block_seconds=block_seconds,train_seconds=float(elapsed),resume_count=resume_count,
                network_construction_seconds=float(build_seconds),initial_state_load_seconds=float(load_seconds),
                max_cuda_memory_allocated_bytes=max(peak_before,int(torch.cuda.max_memory_allocated(dev))) if dev.type=='cuda' else 0,
                max_cuda_memory_reserved_bytes=max(reserved_before,int(torch.cuda.max_memory_reserved(dev))) if dev.type=='cuda' else 0))
        if progress is not None:progress(block+1,blocks,float(elapsed))
    _sync(dev);train_seconds=seconds_before+time.perf_counter()-training_started
    expected_steps=blocks*epochs_per_block*updates;steps=_optimizer_steps(opt)
    if not steps or any(n!=expected_steps for n in steps):raise RuntimeError('Final Adam steps differ from frozen epoch budget')
    source_after=None if initial_state is None else state_dict_hash(initial_state)
    if source_after!=source_before:raise RuntimeError('Caller source state mutated during target learning')
    record=dict(architecture=architecture,network=architecture,output=output,mode=mode,fit_trials=len(features),
        input_shape=list(features.shape[1:]),channels=29,time_samples=512,bands=9 if architecture=='FBCNet' else 1,
        feature_dimensions=256 if architecture=='EEGNet' else 1152,output_width=2 if output=='CE' else 16,
        loss='cross_entropy' if output=='CE' else'binary_cross_entropy_with_logits',optimizer='Adam',learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,seed_init=int(seed_init),seed_batch=int(seed_batch),training_rng_seed=int(seed_batch),device=str(dev),
        blocks_completed=blocks,epochs_per_block=epochs_per_block,epochs_completed=blocks*epochs_per_block,
        optimizer_updates_per_epoch=updates,optimizer_steps_per_parameter=expected_steps,fit_prediction_passes=blocks,
        hard_targets_unchanged=True,hard_targets_sha256=array_hash(target_array),
        finite_gradients_every_optimizer_step=True,parameters_and_buffers_finite_every_block=True,
        fit_outputs_finite_in_unit_interval_every_block=True,fresh_optimizer_state=True,
        initial_fresh_state_sha256=fresh,initial_fresh_backbone_sha256=fresh_backbone,initial_training_state_sha256=initial,
        final_state_sha256=state_hash(model),final_backbone_sha256=backbone_hash(model),
        training_global_RNG_initial_sha256=initial_rng,batch_generator_initial_sha256=generator_initial,
        batch_generator_final_sha256=_generator_hash(generator),source_initial_state_sha256=source_before,
        source_state_after_sha256=source_after,source_state_unchanged=None if initial_state is None else True,
        network_construction_seconds=float(build_seconds),initial_state_load_seconds=float(load_seconds),block_seconds=block_seconds,
        train_seconds=float(train_seconds),total_seconds=float(time.perf_counter()-started),cuda_memory_baseline_bytes=baseline,
        max_cuda_memory_allocated_bytes=max(peak_before,int(torch.cuda.max_memory_allocated(dev))) if dev.type=='cuda' else 0,
        max_cuda_memory_reserved_bytes=max(reserved_before,int(torch.cuda.max_memory_reserved(dev))) if dev.type=='cuda' else 0,
        prediction_metrics_computed=False,test_arrays_accepted=False,engineering_only=engineering_label is not None,
        engineering_label=engineering_label,campaign_eligible=engineering_label is None and(blocks,epochs_per_block)==(10,30),
        checkpoint_schema=CHECKPOINT_SCHEMA,checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        resume_count=resume_count,resumed_from_completed_block=start_block,
        resume_identity_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest())
    arrays=dict(features_fit=fit_features.copy(),logits_fit=logits.copy(),P_fit=probability.copy(),
        Y_used_last=target_array.copy(),y_fit=labels.copy())
    return model,record,arrays
