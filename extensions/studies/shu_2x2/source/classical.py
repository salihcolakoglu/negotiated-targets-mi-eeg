"""Fixed single-band and shared-filterbank CSP–LDA comparator pipelines.

Only the caller's50 first-day training rows fit a pipeline. Prediction applies
frozen filters/classifier only. No data loading, feature search or probability
calibration occurs in this module.
"""
from __future__ import annotations
import copy
import numpy as np
import classifiers_metrics as cm
import filterbank

CONDITIONS=('SINGLE_CSP_LDA','BANK_CSP_LDA')
SCHEMA='shu-fixed-classical-pipeline-v1'


def _input(X,fit=False):
    x=np.asarray(X)
    cm.require(x.dtype==np.float64 and x.ndim==3 and x.shape[1:]==(29,512) and len(x)>0
        and np.isfinite(x).all(),'Classical pipeline expects finite float64 N29x512')
    if fit:cm.require(len(x)==50,'Classical pipelines fit exactly50 training rows; reserved20 cannot fit')
    return x


def configuration(condition):
    cm.require(condition in CONDITIONS,'Unknown fixed classical condition')
    common=dict(fit_trials=50,classes=[0,1],class_counts=[25,25],
        CSP=dict(n_components=6,reg='ledoit_wolf',cov_est='concat',rank='full',norm_trace=False,
            component_order='alternate',component_order_meaning='three largest and three smallest eigenvalues, alternating',
            log=True,transform_into='average_power'),
        LDA=dict(solver='lsqr',shrinkage='auto',priors=[.5,.5]),
        no_feature_selection=True,no_test_fit_or_refit=True,
        feature='MNE log average squared CSP output; per-band six components',
        source_information='Target50 only; unlike transferred networks, no additional source-person training information')
    if condition=='SINGLE_CSP_LDA':
        return dict(condition=condition,feature_count=6,n_bands=1,**common,
            filtering=copy.deepcopy(cm.csp_configuration()),
            pipeline_scope='Fixed8–30Hz Butterworth4/SOS oddpad27 zero-phase +6CSP +LDA')
    return dict(condition=condition,feature_count=54,n_bands=9,**common,
        filtering=copy.deepcopy(filterbank.configuration()),
        pipeline_scope='Exactly the FBCNet shared9band filterbank; independent6CSP per band, concatenate54, no band/feature selection')


def fit_pipeline(X_train50,y_int64,condition='SINGLE_CSP_LDA'):
    cfg=configuration(condition);x=_input(X_train50,fit=True);y=cm.labels(y_int64,50,balanced=25)
    x_before=cm.array_sha256(x);y_before=cm.array_sha256(y)
    if condition=='SINGLE_CSP_LDA':
        pipeline=cm.fit_csp_lda(x,y)
    else:
        from mne.decoding import CSP
        from mne import use_log_level
        bank=filterbank.apply_bank(x)
        cm.require(bank.dtype==np.float64 and bank.shape==(50,9,29,512) and np.isfinite(bank).all(),'Shared bank output mismatch')
        csps=[];parts=[]
        for k in range(9):
            csp=CSP(n_components=6,reg='ledoit_wolf',log=True,cov_est='concat',
                transform_into='average_power',norm_trace=False,rank='full',component_order='alternate')
            with use_log_level('ERROR'):csp.fit(bank[:,k],y)
            part=np.asarray(csp.transform(bank[:,k]),dtype=np.float64)
            cm.require(part.shape==(50,6) and np.isfinite(part).all(),'Invalid bank CSP feature block')
            csps.append(csp);parts.append(part)
        features=np.concatenate(parts,axis=1)
        cm.require(features.shape==(50,54) and features.dtype==np.float64 and np.isfinite(features).all(),'Invalid54-feature bank concatenation')
        pipeline=dict(csps=csps,lda=cm.fit_lda(features,y),fit_n=50,
            fit_features_sha256=cm.array_sha256(features),band_feature_dimensions=[6]*9)
    cm.require(cm.array_sha256(x)==x_before and cm.array_sha256(y)==y_before,'Pipeline fit mutated caller training arrays')
    return dict(schema=SCHEMA,condition=condition,configuration=cfg,fit_n=50,feature_count=cfg['feature_count'],
        fit_X_sha256=x_before,fit_y_sha256=y_before,pipeline=pipeline)


def predict_scores(model,X):
    cm.require(isinstance(model,dict) and model.get('schema')==SCHEMA,'Invalid classical pipeline schema')
    condition=model['condition'];cfg=configuration(condition)
    cm.require(model['configuration']==cfg and model['fit_n']==50 and model['feature_count']==cfg['feature_count'],
        'Classical frozen configuration/fit budget changed')
    x=_input(X);pipe=model['pipeline']
    if condition=='SINGLE_CSP_LDA':scores=cm.csp_lda_scores(pipe,x)
    else:
        cm.require(pipe['fit_n']==50 and len(pipe['csps'])==9 and pipe['band_feature_dimensions']==[6]*9,'Invalid frozen bank pipeline')
        bank=filterbank.apply_bank(x)
        parts=[np.asarray(c.transform(bank[:,k]),dtype=np.float64) for k,c in enumerate(pipe['csps'])]
        features=np.concatenate(parts,axis=1)
        cm.require(features.shape==(len(x),54) and np.isfinite(features).all(),'Invalid frozen54-dimensional features')
        scores=cm.lda_scores(pipe['lda'],features)
    scores=np.asarray(scores)
    cm.require(scores.dtype==np.float64 and scores.shape==(len(x),2) and np.isfinite(scores).all(),'Invalid frozen classical class scores')
    return scores
