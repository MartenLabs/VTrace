import numpy as np


def match_target_loudness(target_signal, reference_signal):
    """
    target_signal의 평균 dBFS를 reference_signal의 평균 dBFS에 맞춤
    """
    def rms_db(signal):
        rms = np.sqrt(np.mean(signal**2)) + 1e-8
        db = 20 * np.log10(rms)
        return db

    ref_db = rms_db(reference_signal)
    target_db = rms_db(target_signal)

    gain_db = ref_db - target_db
    gain_linear = 10 ** (gain_db / 20)

    return target_signal * gain_linear



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

