import librosa
import numpy as np

# ⚠️ Noise Gate는 현재 음질 저하 문제로 사용 안 함
# - smooth=True: 음원 거의 사라짐
# - smooth=False: 중역대 먹먹함
# - 추후 Spectral Gate 개선 필요


# ⚠️ 증폭 및 Noise Gate는 실험 중. 신뢰성 낮음.

def apply_gain(signal, gain_db=2.0):
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear


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


def smoothstep(x):
    return 3 * x**2 - 2 * x**3

def noise_gate(signal, sr, threshold_db=-40, low_cut=300, high_cut=3000, smooth=False, gamma=1.0):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    
    gated = np.zeros_like(signal)
    threshold = 10 ** (threshold_db / 20)  # dB to linear

    for ch in range(signal.shape[1]):
        stft_signal = librosa.stft(signal[:, ch])
        magnitude, phase = np.abs(stft_signal), np.angle(stft_signal)
        
        freqs = librosa.fft_frequencies(sr=sr)
        mask = (freqs >= low_cut) & (freqs <= high_cut)

        if smooth:
            max_mag = np.max(magnitude[mask, :])
            norm_mag = (magnitude[mask, :] - threshold) / (max_mag - threshold + 1e-8)
            norm_mag = np.clip(norm_mag, 0, 1)
            attenuation = (3 * norm_mag**2 - 2 * norm_mag**3) ** gamma
            magnitude[mask, :] *= attenuation
        else:
            magnitude[mask, :] = np.where(magnitude[mask, :] > threshold, magnitude[mask, :], 0)

        gated_stft = magnitude * np.exp(1j * phase)
        gated[:, ch] = librosa.istft(gated_stft, length=signal.shape[0])

    max_val = np.max(np.abs(gated))
    if max_val > 0.99:
        gated = gated / (max_val * 1.01)

    return gated
