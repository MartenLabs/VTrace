import numpy as np
import soundfile as sf
import librosa
from audio_utils.alignment import align_audio, align_signals
from audio_utils.loudness import match_target_loudness, peak_normalize

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

    # 차원/채널 정리
    original, instrumental = align_audio(original, instrumental)

    # (선택) 위상 정렬
    instrumental, lag = align_signals(original, instrumental, max_shift=2000)

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
    
    blended = peak_normalize(blended, headroom_db=1.0)
    blended = match_target_loudness(blended, original)
    sf.write(output_file, blended, target_sr)
    return output_file
