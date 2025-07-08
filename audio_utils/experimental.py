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


def soft_noise_gate(signal, threshold_db=-60.0, softness=0.2):
    """
    - 작은 소리를 부드럽게 줄이되 완전히 자르지 않음
    - softness: 0.1~0.3 사이 추천
    """
    threshold = 10 ** (threshold_db / 20)
    gain = np.clip((np.abs(signal) - threshold) / (threshold * softness), 0.0, 1.0)
    gain = gain ** 1.5  # optional: 감쇠 곡선 조정
    return signal * gain


def normalize_to_target_rms(signal, target_rms_db=-20.0):
    """
    RMS 기반으로 정규화. -20dBFS 정도가 일반적인 vocal 기준.
    """
    rms = np.sqrt(np.mean(signal**2)) + 1e-8
    rms_db = 20 * np.log10(rms)
    gain_db = target_rms_db - rms_db
    gain_linear = 10 ** (gain_db / 20)
    return signal * gain_linear

def safe_loudness_boost(signal, gain_db=3.0, target_peak=0.99, soft_clip=False):
    """
    - signal: np.ndarray (float32, -1.0 ~ 1.0 범위)
    - gain_db: 증폭할 데시벨 (예: 3.0 = +3dB)
    - target_peak: 최대 허용 피크값 (0.99 이하 권장)
    - soft_clip: True일 경우 tanh 기반 소프트 클리핑 사용
    
    기능 순서:
    1. peak normalize (to target_peak)
    2. gain 적용 (gain_db)
    3. clipping or soft clipping
    """
    # Step 1: peak normalize
    max_val = np.max(np.abs(signal)) + 1e-8
    if max_val > 0:
        signal = signal / max_val * target_peak

    # Step 2: gain 적용
    gain_linear = 10 ** (gain_db / 20)
    amplified = signal * gain_linear

    # Step 3: clipping 처리
    if soft_clip:
        # 소프트 클리핑 (부드럽게 눌러줌)
        amplified = np.tanh(amplified / target_peak) * target_peak
    else:
        # 하드 클리핑 (딱 잘라냄)
        amplified = np.clip(amplified, -target_peak, target_peak)

    return amplified


def smoothstep(x):
    return 3 * x**2 - 2 * x**3


def noise_gate(signal, sr, threshold_db=-40, low_cut=300, high_cut=3000, smooth=True, gamma=1.0):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    gated = np.zeros_like(signal)
    threshold = 10 ** (threshold_db / 20)

    for ch in range(signal.shape[1]):
        stft = librosa.stft(signal[:, ch])
        magnitude, phase = np.abs(stft), np.angle(stft)

        freqs = librosa.fft_frequencies(sr=sr)
        mask = (freqs >= low_cut) & (freqs <= high_cut)
        band = magnitude[mask, :]

        if smooth:
            norm_mag = (band - threshold) / (np.max(band) - threshold + 1e-8)
            norm_mag = np.clip(norm_mag, 0, 1)
            attenuation = (3 * norm_mag**2 - 2 * norm_mag**3) ** gamma
            magnitude[mask, :] = band * attenuation
        else:
            magnitude[mask, :] = np.where(band > threshold, band, 0)

        gated_stft = magnitude * np.exp(1j * phase)
        gated[:, ch] = librosa.istft(gated_stft, length=signal.shape[0])

    return gated
