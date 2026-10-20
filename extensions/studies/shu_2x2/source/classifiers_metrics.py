"""Training-only classical heads and held-out class-score calibration.

All data-taking fit APIs enforce50 training trials or20 calibration trials.
No EEG loading, network fitting or test-driven hyperparameter selection occurs
in this module. CSP includes an explicitly different8-30Hz feature pipeline.
"""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'qa/mplconfig'))
import hashlib
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import butter,sosfiltfilt
from scipy.special import expit,logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TEMPERATURE_BOUNDS=(1.,100.)
CSP_SOS=butter(4,[8.,30.],btype='bandpass',fs=128.,output='sos')
CSP_PADLEN=27


def require(ok,message):
    if not bool(ok):raise RuntimeError(message)


def array_sha256(value):
    value=np.ascontiguousarray(value)
    d=hashlib.sha256()
    d.update(str(value.dtype).encode('ascii'))
    d.update(json.dumps(list(value.shape),separators=(',',':')).encode('ascii'))
    d.update(value.tobytes())
    return d.hexdigest()


def labels(y,n,*,balanced=None):
    y=np.asarray(y)
    require(y.dtype==np.int64 and y.shape==(n,) and set(y.tolist())=={0,1},'Expected int64 labels for both classes')
    if balanced is not None:require(np.bincount(y,minlength=2).tolist()==[balanced,balanced],'Wrong class-balanced fit budget')
    return y


def checked_scores(scores):
    scores=np.asarray(scores)
    require(scores.dtype==np.float64 and scores.ndim==2 and scores.shape[1]==2 and len(scores)>0 and np.isfinite(scores).all(),
            'Expected finite float64 two-class scores')
    return scores


def walsh_scores(logits,bits):
    logits,bits=np.asarray(logits),np.asarray(bits)
    require(logits.dtype==bits.dtype==np.float32 and logits.ndim==2 and logits.shape[1]==16
            and bits.shape==logits.shape and len(logits)>0 and np.isfinite(logits).all() and np.isfinite(bits).all(),
            'Expected retained float32 Walsh logits/bits')
    codes=np.asarray([[1.]*16,[1.,0.]*8],dtype=np.float64)
    p=expit(logits.astype(np.float64))
    require(np.allclose(p,bits,rtol=0,atol=1e-7),'Sigmoid/logit mismatch')
    clipped=np.clip(p,1e-7,1-1e-7)
    scores=np.log(clipped)@codes.T+np.log1p(-clipped)@(1-codes.T)
    pred=np.square(bits[:,None,:]-codes.astype(np.float32)[None,:,:]).sum(axis=2).argmin(axis=1).astype(np.int64)
    return scores,pred


def fit_lda(features,y):
    f=np.asarray(features,dtype=np.float64)
    require(f.ndim==2 and f.shape[0]==50 and f.shape[1] in (6,54) and np.isfinite(f).all(),
            'LDA fits exactly50 training rows and6or256 features')
    y=labels(y,50,balanced=25)
    model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.asarray([.5,.5]))
    model.fit(f,y)
    require(np.array_equal(model.classes_,[0,1]),'Unexpected LDA class order')
    model._study_fit_n=50
    model._study_fit_features_sha256=array_sha256(f)
    model._study_fit_y_sha256=array_sha256(y)
    model._study_feature_count=f.shape[1]
    return model


def lda_scores(model,features):
    f=np.asarray(features,dtype=np.float64)
    require(f.ndim==2 and f.shape[1]==model._study_feature_count and np.isfinite(f).all(),'Invalid frozen-head features')
    require(model._study_fit_n==50 and model.solver=='lsqr' and model.shrinkage=='auto'
            and np.array_equal(model.classes_,[0,1]),'Unexpected LDA configuration')
    margin=np.asarray(model.decision_function(f),dtype=np.float64)
    require(margin.shape==(len(f),) and np.isfinite(margin).all(),'Nonfinite LDA margin')
    # Symmetric two-class scores with the exact sklearn binary margin.
    scores=np.column_stack((-.5*margin,.5*margin)).astype(np.float64)
    require(np.array_equal(scores.argmax(axis=1),model.predict(f)),'LDA score/predict mismatch')
    return scores


def bandpass_csp(X):
    x=np.asarray(X,dtype=np.float64)
    require(x.ndim==3 and x.shape[1:]==(29,512) and len(x)>0 and np.isfinite(x).all(),'Invalid29x512 CSP input')
    out=sosfiltfilt(CSP_SOS,x,axis=-1,padtype='odd',padlen=CSP_PADLEN)
    require(np.isfinite(out).all(),'Nonfinite fixed CSP filter output')
    return out


def fit_csp_lda(X,y):
    from mne.decoding import CSP
    from mne import use_log_level
    x=np.asarray(X,dtype=np.float64)
    require(x.shape==(50,29,512),'CSP fits exactly50 target training trials')
    y=labels(y,50,balanced=25)
    filtered=bandpass_csp(x)
    csp=CSP(n_components=6,reg='ledoit_wolf',log=True,cov_est='concat',
            transform_into='average_power',norm_trace=False,rank='full',component_order='alternate')
    with use_log_level('ERROR'):
        csp.fit(filtered,y)
    features=np.asarray(csp.transform(filtered),dtype=np.float64)
    require(features.shape==(50,6) and np.isfinite(features).all(),'Invalid two-ended CSP log-power features')
    lda=fit_lda(features,y)
    return dict(csp=csp,lda=lda,fit_n=50,fit_X_sha256=array_sha256(x),fit_y_sha256=array_sha256(y),
                configuration=csp_configuration())


def csp_lda_scores(model,X):
    require(model['fit_n']==50 and model['configuration']==csp_configuration(),'Wrong frozen CSP model configuration')
    features=np.asarray(model['csp'].transform(bandpass_csp(X)),dtype=np.float64)
    return lda_scores(model['lda'],features)


def csp_configuration():
    return dict(band_hz=[8.,30.],fs=128.,order=4,implementation='scipy.signal.butter SOS + sosfiltfilt per trial',
        sos=CSP_SOS.tolist(),padtype='odd',padlen=CSP_PADLEN,n_components=6,reg='ledoit_wolf',
        covariance='concat',rank='full',norm_trace=False,component_order='alternate',
        component_order_meaning='largest,smallest,second-largest,second-smallest,third-largest,third-smallest eigenvalues',
        feature='MNE log average squared CSP output (average_power), not an added test normalization',
        lda_solver='lsqr',lda_shrinkage='auto',lda_priors=[.5,.5],
        comparison_scope='Different band/spatial-feature pipeline from CNN; CSP-versus-CNN is not a pure classifier-head contrast.')


def normalized_log_probabilities(scores,T=1.):
    scores=checked_scores(scores)
    require(np.isfinite(T) and float(T)>0,'Temperature must be finite positive')
    scaled=scores/float(T)
    require(np.isfinite(scaled).all(),'Temperature-scaled scores overflowed')
    return scaled-logsumexp(scaled,axis=1,keepdims=True)


def fit_temperature(scores,y):
    scores=checked_scores(scores)
    require(len(scores)==20,'Temperature fits only the reserved20 Day1 rows')
    y=labels(y,20,balanced=10)
    def objective(log_t):
        lp=normalized_log_probabilities(scores,np.exp(log_t))
        return float(-lp[np.arange(20),y].mean())
    lo,hi=TEMPERATURE_BOUNDS
    fit=minimize_scalar(objective,bounds=(np.log(lo),np.log(hi)),method='bounded',
                        options=dict(xatol=1e-10,maxiter=1000))
    require(fit.success and np.isfinite(fit.fun),'Temperature optimizer failed')
    candidates=[(lo,objective(np.log(lo))),(hi,objective(np.log(hi))),(1.,objective(0.)),
                (float(np.exp(fit.x)),float(fit.fun))]
    best=min(candidates,key=lambda v:(v[1],abs(np.log(v[0])),v[0]))
    T=best[0]
    require(lo<=T<=hi and best[1]<=objective(0.)+1e-12,'Invalid held-out calibration selection')
    return dict(temperature=T,at_lower_bound=bool(T==lo),at_upper_bound=bool(T==hi),
        n_calibration=20,calibration_class_counts=[10,10],source='reserved20Day1',
        bounds=[lo,hi],optimizer='bounded scalar in log temperature; endpoints andT1 explicitly evaluated',
        optimizer_success=bool(fit.success),optimizer_nfev=int(fit.nfev),nll_before=objective(0.),nll_after=best[1],
        scores_sha256=array_sha256(scores),labels_sha256=array_sha256(y),
        applied_to='two-class scores only; original decision array retained',
        no_test_labels_used=True)


def probability_metrics(scores,y,original_pred=None,T=1.):
    scores=checked_scores(scores);y=labels(y,len(scores))
    lp=normalized_log_probabilities(scores,T);p=np.exp(lp)
    # Positive temperature preserves score order mathematically.  Exp(logp)
    # can round nearly equal probabilities to identical floats; do not turn
    # that numerical rounding into a changed class decision.
    posterior_pred=scores.argmax(axis=1)
    if original_pred is None:original_pred=scores.argmax(axis=1).astype(np.int64)
    original_pred=np.asarray(original_pred)
    require(original_pred.dtype==np.int64 and original_pred.shape==y.shape and np.isin(original_pred,[0,1]).all(),'Invalid retained original decision')
    confidence=p.max(axis=1);correct=posterior_pred==y;ece=0.;edges=np.linspace(0,1,16)
    for k in range(15):
        mask=((confidence>=edges[k]) if k==0 else (confidence>edges[k]))&(confidence<=edges[k+1])
        if mask.any():ece+=mask.mean()*abs(np.mean(correct[mask])-confidence[mask].mean())
    return dict(accuracy=float(np.mean(original_pred==y)),
        balanced_accuracy=float(np.mean([np.mean(original_pred[y==c]==c) for c in (0,1)])),
        recall_left=float(np.mean(original_pred[y==0]==0)),recall_right=float(np.mean(original_pred[y==1]==1)),
        nll=float(-lp[np.arange(len(y)),y].mean()),ece15=float(ece),
        brier=float(np.square(p-np.eye(2)[y]).sum(axis=1).mean()),
        posterior_accuracy=float(np.mean(posterior_pred==y)),
        decoder_posterior_disagreement_fraction=float(np.mean(original_pred!=posterior_pred)),
        mean_max_probability=float(confidence.mean()),n=int(len(y)),
        n_class0=int(np.sum(y==0)),n_class1=int(np.sum(y==1)),n_correct=int(np.sum(original_pred==y)),
        n_posterior_correct=int(np.sum(posterior_pred==y)),n_disagreement=int(np.sum(original_pred!=posterior_pred)))

