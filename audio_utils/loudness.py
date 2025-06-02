import numpy as np

def simple_rms_match(target_signal, reference_signal):
    ref_rms = np.sqrt(np.mean(reference_signal**2)) + 1e-8
    target_rms = np.sqrt(np.mean(target_signal**2)) + 1e-8

    scale = ref_rms / target_rms
    return target_signal * scale


def peak_normalize(signal, headroom_db=1.0):
    """
    신호의 최대 피크를 headroom dBFS 이내로 정규화
    (예: headroom_db=1.0 이면 최대 피크 -1dBFS)
    """
    max_val = np.max(np.abs(signal)) + 1e-8
    if max_val > 0:
        target_linear = 10 ** (-headroom_db / 20)
        return signal * (target_linear / max_val)
    return signal

