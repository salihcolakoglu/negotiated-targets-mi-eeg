"""Singleton W4 campaign; source/IEEE files stay immutable."""
import os
for _key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,importlib.util,json,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import campaign as c


def monitor():
    spec=importlib.util.spec_from_file_location('lee_w4_safe_monitor',ROOT/'ops/monitor_campaign-V1.0.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main(protocol,ps,resume=False,validate_only=False):
    p=c.validate_protocol(protocol,ps,check_runtime=not validate_only)
    if validate_only:return 'VALIDATED_METADATA_ONLY'
    mon=monitor();lock=ROOT/'campaign/RUNNING.lock';statuspath=ROOT/'campaign/STATUS.json'
    lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        old=json.loads(mon.shared_read(lock))
        c.require(not mon.alive(old['pid']),'Existing campaign PID alive; never duplicate the process')
        c.require(resume and old['protocol_sha256']==ps,'Only explicit same-protocol dead-lock resume')
        c.archive_orphan(lock,'Confirmed old PID dead; same-protocol explicit resume')
    lease=dict(pid=os.getpid(),protocol_sha256=ps,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with lock.open('x',encoding='utf8')as f:json.dump(lease,f)
    def counts():
        snap=mon.snapshot();return dict(fits=snap['completed_fits'],tests=snap['completed_tests'],
                                        preprocessors=snap['completed_preprocessors'])
    def status(**fields):
        c.atomic_json(statuspath,dict(lease,status='RUNNING',counts=counts(),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields))
    try:
        status(stage='VALIDATED_START')
        state=c.Campaign(p,ps,status).run()
        if state=='STOPPED_AFTER_CURRENT_FIT':
            c.atomic_json(statuspath,dict(lease,status=state,counts=counts()));return state
        status(stage='FINAL_ANALYSIS')
        argv=[sys.executable,'-X','utf8','-B',str(ROOT/'analysis/final_analysis.py'),
              '--protocol',str(c.PROTOCOL),'--protocol-sha256',ps]
        subprocess.run(argv,cwd=str(ROOT),check=True,env=os.environ.copy())
        resultpath=ROOT/'final_analysis/ANCHOR_GATE_RESULTS-V1.0.json';result=c.read(resultpath)
        c.require(result['status']=='COMPLETE_ANCHOR_GATE_ABLATION_ANALYSIS_VERIFIED','Final W4 analysis not verified')
        complete=dict(status='COMPLETE',protocol_sha256=ps,counts=c.COUNTS,
            model_freeze_sha256=c.sha(c.MODEL_FREEZE),test_completion_sha256=c.sha(c.TEST_COMPLETION),
            results_sha256=c.sha(resultpath),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            no_test_driven_changes=True)
        target=ROOT/'ops/CAMPAIGN_COMPLETION-V1.0.json'
        if target.exists():
            prior=c.read(target);c.require(all(prior.get(k)==v for k,v in complete.items()if k!='completed_utc'),
                                         'Completed resume changed immutable scientific completion')
        else:c.atomic_json(target,complete,immutable=True)
        c.atomic_json(statuspath,dict(lease,status='COMPLETE',stage='VERIFIED_COMPLETE',counts=counts()))
        return 'COMPLETE'
    except BaseException as error:
        failure=dict(lease,status='FAILED',counts=counts(),error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
        c.atomic_json(ROOT/f'ops/CAMPAIGN_FAILURE-{stamp}-V1.0.json',failure,immutable=True)
        c.atomic_json(statuspath,failure);raise
    finally:
        if lock.exists():lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    print(main(args.protocol,args.protocol_sha256,args.resume,args.validate_only))
