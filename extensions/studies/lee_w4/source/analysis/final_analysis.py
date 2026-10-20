"""Complete-cohort analysis of the prospective W4 mechanism control.

Old Lee results are historical paired comparators, never extra participants.
No numeric prediction archive is decoded before all new gates are verified.
"""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="2"
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]

def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text("utf-8"))
def identities(condition="W4"):
    return {f"subject_{s:02d}_train__{condition}__seed{k}":(s,k)
            for s in range(1,55) for k in range(1,6)}
def local(rel):
    p=(ROOT/rel).resolve()
    require(p.is_relative_to(ROOT),f"Path outside study: {rel}")
    return p
def check_files(files):
    for rel,h in files.items():require(sha(local(rel))==h,f"Changed file: {rel}")
def immutable(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,f"Analysis bytes differ: {path}")
    else:
        with path.open("xb") as f:f.write(data)
def encoded(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")

def codebook():
    # Literal binary Sylvester row0/row1, matches the preserved source row order.
    return np.array([[1]*16,[1,0]*8],dtype=np.float32)
def metric(P,y):
    require(P.dtype==np.float32 and P.shape==(100,16),"Exact saved output dtype/shape")
    require(y.dtype==np.int64 and y.shape==(100,),"Exact saved labels")
    require(np.isfinite(P).all() and np.all((P>=0)&(P<=1)),"Finite sigmoid values")
    require(np.unique(y,return_counts=True)[1].tolist()==[50,50] and set(y)=={0,1},"Balanced binary test")
    C=codebook().astype(np.float64);p=P.astype(np.float64)
    pred=((p[:,None,:]-C[None,:,:])**2).sum(2).argmin(1)
    pc=np.clip(p,1e-7,1-1e-7)
    logp=np.log(pc)@C.T+np.log(1-pc)@(1-C.T)
    logp-=logp.max(1,keepdims=True)
    post=np.exp(logp);post/=post.sum(1,keepdims=True)
    conf=post.max(1);post_pred=post.argmax(1)
    ece=0.;edges=np.linspace(0,1,16)
    for i in range(15):
        mask=(conf>=edges[i])&(conf<=edges[i+1]) if i==0 else (conf>edges[i])&(conf<=edges[i+1])
        if mask.any():ece+=mask.mean()*abs((post_pred[mask]==y[mask]).mean()-conf[mask].mean())
    Y=C[y]
    return dict(accuracy=float((pred==y).mean()),
                nll_clipped=float(-np.log(np.clip(post[np.arange(100),y],1e-12,1)).mean()),
                ece15=float(ece),
                bce=float(-(Y*np.log(pc)+(1-Y)*np.log(1-pc)).mean()),
                brier_sum=float(np.mean(np.sum((post-np.eye(2)[y])**2,axis=1))),
                mean_confidence=float(conf.mean()),
                posterior_accuracy=float((post_pred==y).mean()))
def compare_metric(r,m):
    for field,key in (("acc_test","accuracy"),("nll_test","nll_clipped"),
                      ("ece_test","ece15"),("bce_test","bce")):
        if field not in r and field=="nll_test":field="nll_test_clipped"
        require(field in r,f"Receipt missing {field}")
        require(abs(float(r[field])-m[key])<1e-11,f"Metric mismatch {field}")

def paired(values):
    d=np.asarray(values,dtype=np.float64)
    require(d.shape==(54,) and np.isfinite(d).all(),"54 person differences required")
    mean=float(d.mean());sd=float(d.std(ddof=1))
    if sd==0:
        t=0. if mean==0 else None
        p=1. if mean==0 else 0.
        ci=[mean,mean]
        dz=None if mean else 0.
    else:
        t=mean/(sd/math.sqrt(54))
        p=float(2*stats.t.sf(abs(t),53))
        half=float(stats.t.ppf(.975,53))*sd/math.sqrt(54)
        ci=[mean-half,mean+half];dz=mean/sd
    return dict(mean_difference_pp=mean,pointwise_95ci_pp=ci,sd_difference_pp=sd,
                t=t,df=53,p_unadjusted=p,d_z=dz,
                participants_improved=int((d>0).sum()),participants_worse=int((d<0).sum()),
                participants_tied=int((d==0).sum()),n=54)
def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind="stable");out=np.empty_like(p);running=0.
    for i,j in enumerate(order):
        running=max(running,float((len(p)-i)*p[j]));out[j]=min(1.,running)
    return out.tolist()
def csv_bytes(rows):
    out=io.StringIO(newline="");w=csv.DictWriter(out,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows);return out.getvalue().encode("utf-8")

def run(protocol,expected):
    protocol=Path(protocol).resolve();require(sha(protocol)==expected,"Protocol SHA")
    p=load(protocol)
    require(p["status"]=="FROZEN_BEFORE_ANCHOR_GATE_ABLATION","Prospective protocol status")
    require(p["counts"]==dict(fits=270,preprocessors=54,tests=270),"Fixed counts")
    require(p["analysis"]["contrasts"]==["W2-W4","W4-W1"],"Fixed two-contrast family")
    for rel,h in p["code_sha256"].items():require(sha(local(rel))==h,f"Code changed: {rel}")
    bpath=ROOT/"data/SOURCE_BINDINGS-V1.0.json"
    bsha=sha(bpath);require(bsha==p["source_bindings_sha256"],"Source binding SHA")
    b=load(bpath)
    for path,h in b["readonly_sha256"].items():require(sha(path)==h,f"Source changed: {path}")
    for rel,h in p.get("qa_sha256",{}).items():require(sha(local(rel))==h,f"QA changed: {rel}")
    freeze=load(ROOT/"campaign/MODEL_FREEZE-V1.0.json")
    freeze_sha=sha(ROOT/"campaign/MODEL_FREEZE-V1.0.json")
    require(freeze["status"]=="ALL270_FITS_FROZEN_BEFORE_DAY2","Complete model gate")
    require(freeze["protocol_sha256"]==expected,"Model gate protocol")
    require(freeze["counts"]==dict(fits=270,preprocessors=54,artifacts=864),"864 model artifacts")
    require(len(freeze["files"])==864,"Unique model artifact count");check_files(freeze["files"])
    tc=load(ROOT/"campaign/TEST_COMPLETION-V1.0.json")
    require(tc["status"]=="TESTS_COMPLETE" and tc["protocol_sha256"]==expected,"Complete test gate")
    require(tc["model_freeze_sha256"]==freeze_sha,"Test/model gate link")
    require(tc["counts"]==dict(tests=270,artifacts=540) and len(tc["files"])==540,"Complete540 gate")
    check_files(tc["files"])
    ids=identities()
    fits={q.stem:load(q) for q in (ROOT/"campaign/fit_records").glob("*.json")}
    tests={q.stem:load(q) for q in (ROOT/"campaign/test_records").glob("*.json")}
    require(set(fits)==set(ids),"Exactly270 unique fit identities")
    require(set(tests)=={fid+"_day2" for fid in ids},"Exactly270 unique test identities")
    expect_frozen=set();expect_test=set()
    for fid,(subject,seed) in ids.items():
        f=fits[fid];t=tests[fid+"_day2"]
        require(f["id"]==fid and f["subject"]==subject and f["seed"]==seed,"Fit identity")
        require(f["condition"]=="W4" and f["epochs_completed"]==300,"Fixed W4 budget")
        require(f["protocol_sha256"]==expected and f["source_bindings_sha256"]==bsha,"Fit source links")
        require(t["id"]==fid+"_day2" and t["fit_id"]==fid,"Test identity")
        require(t["protocol_sha256"]==expected and t["model_freeze_sha256"]==freeze_sha,"Test timing gate")
        require(t["source_fit_sha256"]==sha(ROOT/f"campaign/fit_records/{fid}.json"),"Test fit binding")
        require(sha(local(t["predictions_path"]))==t["prediction_sha256"],"Prediction binding")
        require(f"campaign/fit_records/{fid}.json" in freeze["files"],"Fit receipt in model gate")
        for v in f["files"].values():
            require(sha(local(v["path"]))==v["sha256"],"Fit artifact")
            expect_frozen.add(v["path"])
        expect_frozen.add(f"campaign/fit_records/{fid}.json")
        expect_test.update((f"campaign/test_records/{fid}_day2.json",t["predictions_path"]))
    expect_frozen.update(f"data/preprocessors/subject_{s:02d}.json" for s in range(1,55))
    require(set(freeze["files"])==expect_frozen,"Exact864 model union")
    require(set(tc["files"])==expect_test,"Exact540 test union")
    # Only now open any result arrays, including the historical comparators.
    rows=[];label_order={}
    for condition in ("W1","W2","W4"):
        for fid,(subject,seed) in identities(condition).items():
            if condition=="W4":
                receipt=tests[fid+"_day2"];pred=local(receipt["predictions_path"]);f=fits[fid]
                with np.load(pred,allow_pickle=False) as z:
                    require(set(z.files)=={"P_test_float32","y_test_int64"},"New test members")
                    y=z["y_test_int64"]
                    m=metric(z["P_test_float32"],y)
            else:
                historical=b["historical"][fid];receipt=load(historical["record"]["path"]);f=receipt
                with np.load(historical["prediction"]["path"],allow_pickle=False) as z:
                    y=z["y_test"]
                    m=metric(z["P_test"],y)
            if subject not in label_order:label_order[subject]=y.copy()
            require(np.array_equal(y,label_order[subject]),"Paired condition/seed test-label order")
            compare_metric(receipt,m)
            # New trainer keeps original core's train record fields at the top level.
            rows.append(dict(subject=subject,seed=seed,condition=condition,**m,
                             train_accuracy=float(f["acc_train"]),
                             target_displacement=float(f["last_target_dist"]),
                             train_seconds=float(f["train_seconds"])))
    metrics=[k for k in rows[0] if k not in ("subject","seed","condition")]
    people=[]
    for s in range(1,55):
        for cond in ("W1","W2","W4"):
            group=[r for r in rows if r["subject"]==s and r["condition"]==cond]
            require(len(group)==5,"Five paired seeds per person")
            people.append(dict(subject=s,condition=cond,**{k:float(np.mean([r[k] for r in group])) for k in metrics}))
    summaries=[dict(condition=c,n_participants=54,starts_per_participant=5,
                    **{k:float(np.mean([r[k] for r in people if r["condition"]==c])) for k in metrics})
               for c in ("W1","W2","W4")]
    indexed={(r["subject"],r["condition"]):r for r in people}
    contrasts=[];differences=[]
    for name,left,right in (("W2-W4","W2","W4"),("W4-W1","W4","W1")):
        ds=[round(100*(indexed[(s,left)]["accuracy"]-indexed[(s,right)]["accuracy"]),10) for s in range(1,55)]
        contrasts.append(dict(contrast=name,primary=name=="W2-W4",**paired(ds)))
        for s in range(1,55):
            differences.append(dict(subject=s,contrast=name,accuracy_difference_pp=ds[s-1],
                                    **{k+"_difference":indexed[(s,left)][k]-indexed[(s,right)][k]
                                       for k in metrics if k!="accuracy"}))
    for c,padj in zip(contrasts,holm([c["p_unadjusted"] for c in contrasts])):c["p_holm_two"]=padj
    results=dict(status="COMPLETE_ANCHOR_GATE_ABLATION_ANALYSIS_VERIFIED",
                 protocol_sha256=expected,source_bindings_sha256=bsha,model_freeze_sha256=freeze_sha,
                 test_completion_sha256=sha(ROOT/"campaign/TEST_COMPLETION-V1.0.json"),
                 counts=dict(new_fits=270,new_tests=270,historical_comparators=540,
                             model_artifacts=864,test_artifacts=540,participant_means=162),
                 unit="54 people; each condition is averaged over five paired seeds",
                 scope="Exploratory mechanism control after Lee results had been examined; original five-contrast family unchanged",
                 metrics_note="Historical bit clipping1e-7 and class-posterior floor1e-12 retained. Native Walsh decision uses original float64 Euclidean metric; training gates remain float32.",
                 uncertainty_note="Pointwise95% intervals, not multiplicity-adjusted intervals. Holm family contains exactly two new accuracy contrasts.",
                 summaries=summaries,contrasts=contrasts)
    out=ROOT/"final_analysis"
    files={
        "ANCHOR_GATE_RESULTS-V1.0.json":encoded(results),
        "run_metrics.csv":csv_bytes(rows),"participant_means.csv":csv_bytes(people),
        "condition_summary.csv":csv_bytes(summaries),"participant_differences.csv":csv_bytes(differences),
        "two_contrasts_holm.json":encoded(contrasts)}
    for name,raw in files.items():immutable(out/name,raw)
    report={"status":"COMPLETE_ANCHOR_GATE_ANALYSIS","protocol_sha256":expected,
            "files":{n:sha(out/n) for n in files},"protected_IEEE_files":len(b["protected_files"])}
    immutable(out/"ANALYSIS_COMPLETION-V1.0.json",encoded(report))
    print(json.dumps({"status":results["status"],"counts":results["counts"]}))
    return results
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",required=True);ap.add_argument("--protocol-sha256",required=True)
    a=ap.parse_args();run(a.protocol,a.protocol_sha256)
