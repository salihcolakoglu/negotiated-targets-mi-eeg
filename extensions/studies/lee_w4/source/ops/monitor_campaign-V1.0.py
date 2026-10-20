"""Operational SHARE_READ|WRITE|DELETE snapshot; never opens numeric arrays."""
import ctypes,datetime,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def shared_read(path):
    path=Path(path)
    if os.name!='nt':return path.read_text(encoding='utf8')
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    k.CreateFileW.restype=w.HANDLE
    k.ReadFile.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD),ctypes.c_void_p]
    k.CloseHandle.argtypes=[w.HANDLE]
    handle=k.CreateFileW(str(path),0x80000000,7,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    chunks=[]
    try:
        while True:
            buf=ctypes.create_string_buffer(65536);n=w.DWORD()
            if not k.ReadFile(handle,buf,65536,ctypes.byref(n),None):raise ctypes.WinError(ctypes.get_last_error())
            if not n.value:break
            chunks.append(buf.raw[:n.value])
    finally:k.CloseHandle(handle)
    return b''.join(chunks).decode('utf8')

def alive(pid):
    if type(pid)is not int or pid<=0:raise RuntimeError('Lock PID must be a positive integer')
    if os.name!='nt':
        try:os.kill(int(pid),0);return True
        except ProcessLookupError:return False
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
    k.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)];k.CloseHandle.argtypes=[w.HANDLE]
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:
        error=ctypes.get_last_error()
        if error==87:return False
        raise ctypes.WinError(error)
    code=w.DWORD()
    try:
        if not k.GetExitCodeProcess(h,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:k.CloseHandle(h)

def read_optional(path):
    try:return json.loads(shared_read(path))
    except FileNotFoundError:return None
    except (ValueError,OSError) as e:return {'snapshot_read_error':str(e)}

def snapshot():
    status=read_optional(ROOT/'campaign/STATUS.json');lock=read_optional(ROOT/'campaign/RUNNING.lock')
    expected={f'subject_{s:02d}_train__W4__seed{k}'for s in range(1,55)for k in range(1,6)}
    ids=set();tests=set();errors=[]
    for path in (ROOT/'campaign/fit_records').glob('*.json'):
        r=read_optional(path)
        if not r or 'snapshot_read_error' in r:continue
        if r.get('campaign_eligible') is True and r.get('fit_id')in expected:ids.add(r['fit_id'])
    for path in (ROOT/'campaign/test_records').glob('*.json'):
        r=read_optional(path)
        if r and r.get('id')in {key+'_day2'for key in expected}:tests.add(r['id'])
    logs=[]
    for path in sorted((ROOT/'logs/campaign').glob('*'),key=lambda x:x.stat().st_mtime,reverse=True)[:2]:
        try:logs.append({'path':str(path),'tail':shared_read(path)[-4000:]})
        except OSError as e:logs.append({'path':str(path),'read_error':str(e)})
    try:live=bool(lock and lock.get('pid') and alive(lock['pid']))
    except (OSError,RuntimeError) as e:live=None;errors.append('PID liveness unavailable: '+str(e))
    if status and status.get('status')=='FAILED':errors.append(status.get('error'))
    if lock and live is False:errors.append('Lock PID is not alive')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat()
    return dict(control_time_Europe_Istanbul=now,status=status,lock=lock,lock_PID_alive=live,
        completed_fits=len(ids),planned_fits=270,fit_percent=100*len(ids)/270,
        completed_preprocessors=len(list((ROOT/'data/preprocessors').glob('subject_*.json'))),planned_preprocessors=54,
        completed_tests=len(tests),planned_tests=270,
        model_freeze_present=(ROOT/'campaign/MODEL_FREEZE-V1.0.json').exists(),
        completion=read_optional(ROOT/'ops/CAMPAIGN_COMPLETION-V1.0.json'),latest_logs=logs,current_errors=errors)

if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
