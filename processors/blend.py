import numpy as np
import soundfile as sf
import librosa
from utils.audio_utils import normalize_audio

def blend_audio_tracks(original_file, instrumental_file, output_file, blend_ratio=0.1, blend_mode='linear'):

    original, sr_orig = sf.read(original_file)
    instrumental, sr_inst = sf.read(instrumental_file)

    target_sr = min(sr_orig, sr_inst)
    if sr_orig != target_sr:
        original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=target_sr).T
        sr_orig = target_sr
    
    if sr_inst != target_sr:
        instrumental = librosa.resample(instrumental.T, orig_sr=sr_inst, target_sr=target_sr).T
        sr_inst = target_sr

    min_length = min(len(original), len(instrumental))
    original = original[:min_length]
    instrumental = instrumental[:min_length]

    if original.ndim == 1:
        original = np.column_stack((original, original))
    if instrumental.ndim == 1:
        instrumental = np.column_stack((instrumental, instrumental))
    if original.shape[1] != instrumental.shape[1]:
        target_channels = original.shape[1]
        instrumental = np.tile(instrumental[:, 0:1], (1, target_channels))

    # 블렌드 가중치 계산
    if blend_mode == 'linear':
        w_orig = blend_ratio
    elif blend_mode == 'exp':
        w_orig = np.exp(-blend_ratio)
    elif blend_mode == 'log':
        w_orig = np.log1p(blend_ratio)
    elif blend_mode == 'power':
        gamma = 2  # 기본값
        w_orig = blend_ratio ** gamma
    else:
        raise ValueError(f"지원하지 않는 blend_mode: {blend_mode}")

    w_inst = 1 - w_orig


    blended = (original * w_orig) + (instrumental * w_inst)

    max_val = np.max(np.abs(blended))
    if max_val > 0.99:
        blended = blended / (max_val * 1.01)

    blended_norm = normalize_audio(blended)

    sf.write(output_file, blended_norm, target_sr)
    return output_file
