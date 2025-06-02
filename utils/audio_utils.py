import numpy as np
import librosa

def match_target_amplitude(target_signal, reference_signal, eps=1e-8):
    """
    target_signal의 RMS를 reference_signal의 RMS에 맞춰 스케일링
    """
    ref_rms = np.sqrt(np.mean(reference_signal ** 2))
    target_rms = np.sqrt(np.mean(target_signal ** 2)) + eps

    scale = ref_rms / target_rms
    return target_signal * scale


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




def normalize_audio(signal, method="peak", target_db=-1.0, headroom_db=1.0):

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    if method == "peak":
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            headroom = 10 ** (-headroom_db / 20)
            return signal / max_val * headroom
        return signal
    elif method == "rms":
        rms = np.sqrt(np.mean(signal**2))
        if rms > 0:
            target_linear = 10 ** (target_db / 20)  # dB to linear
            return signal * (target_linear / rms)
        return signal
    else:
        raise ValueError(f"Unknown method: {method}")

