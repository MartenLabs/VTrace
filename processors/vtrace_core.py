import numpy as np
from audio_utils.alignment import align_audio, align_signals

def blend_audio_tracks(original_audio, instrumental_audio, sample_rate, blend_ratio=0.1, blend_mode='linear'):
    original_audio, instrumental_audio = align_audio(original_audio, instrumental_audio)
    instrumental_audio, lag = align_signals(original_audio, instrumental_audio, max_shift=2000)

    if blend_mode == 'linear':
        w_orig = blend_ratio
    elif blend_mode == 'exp':
        w_orig = np.exp(-blend_ratio)
    elif blend_mode == 'log':
        w_orig = np.log1p(blend_ratio)
    elif blend_mode == 'power':
        w_orig = blend_ratio ** 2
    else:
        raise ValueError(f"지원하지 않는 blend_mode: {blend_mode}")

    w_inst = 1 - w_orig
    blended_audio = (original_audio * w_orig) + (instrumental_audio * w_inst)
    return blended_audio, sample_rate

def residual_subtraction(original_audio, blend_audio, sample_rate):
    original_audio, blend_audio = align_audio(original_audio, blend_audio)
    blend_aligned, lag = align_signals(original_audio, blend_audio, max_shift=2000)
    residual_vocal = original_audio - blend_aligned
    return residual_vocal, sample_rate



def process_phase_cancel(original_audio, residual_vocal, sample_rate):
    original_audio, residual_vocal = align_audio(original_audio, residual_vocal)
    residual_vocal, lag = align_signals(original_audio, residual_vocal, max_shift=2000)
    instrumental = original_audio - residual_vocal
    return instrumental, sample_rate



