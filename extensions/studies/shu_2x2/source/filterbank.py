"""Fixed4–40Hz ChebyshevII bank; per-trial float64 SOS forward/backward.

Explicit adaptation of author filterBank: use returned cheb2ord critical Wn,
SOS numerics, oddpad27 and two-pass application rather than default lfilter.
Single-pass specifications3dB pass/30dB stop; two-pass6dB/60dB at edges.
"""
import numpy as np
from scipy.signal import cheb2ord,cheby2,sosfiltfilt

FS=128.
BANDS=tuple((float(lo),float(lo+4)) for lo in range(4,40,4))
PADLEN=27


def _design():
    filters=[]; orders=[]; critical=[]
    for lo,hi in BANDS:
        order,wn=cheb2ord([lo,hi],[lo-2,hi+2],3,30,fs=FS)
        filters.append(cheby2(order,30,wn,btype='bandpass',fs=FS,output='sos'))
        orders.append(int(order)); critical.append(np.asarray(wn).tolist())
    return tuple(filters),orders,critical


SOS,ORDERS,CRITICAL=_design()


def configuration():
    return {'fs':FS,'bands_hz':[list(b) for b in BANDS],'transition_hz':2.,
            'single_pass_passband_loss_db':3.,'single_pass_stop_attenuation_db':30.,
            'two_pass_edge_passband_loss_db':6.,'two_pass_stop_attenuation_db':60.,
            'implementation':'cheb2ord Wn -> cheby2 SOS -> sosfiltfilt pertrial',
            'padtype':'odd','padlen':PADLEN,'orders':ORDERS,'critical_Wn_hz':CRITICAL,
            'sos':[s.tolist() for s in SOS],'dtype':'float64',
            'source_author_default_is_causal_lfilter':True,
            'adaptation_not_byte_identical_author_preprocessing':True}


def apply_bank(X):
    x=np.asarray(X)
    if x.ndim!=3 or tuple(x.shape[1:])!=(29,512) or len(x)==0 or not np.isfinite(x).all():
        raise ValueError('Bank input must be finite N29x512')
    x=np.asarray(x,dtype=np.float64)
    out=np.stack([sosfiltfilt(s,x,axis=-1,padtype='odd',padlen=PADLEN) for s in SOS],axis=1)
    if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite bank result')
    return out
