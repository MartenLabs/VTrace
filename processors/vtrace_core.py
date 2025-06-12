import numpy as np
from audio_utils.alignment import align_audio, align_signals

def blend_audio_tracks(original_audio, instrumental_audio, blend_ratio: float = 0.1, blend_mode: str = 'linear'
):
    """
    보컬 감쇠를 위한 오디오 블렌딩 함수

    Parameters:
    - original_audio: 원본 오디오 (numpy array)
    - instrumental_audio: 무보컬 오디오 (numpy array)
    - sample_rate: 샘플레이트
    - blend_ratio: 보컬 감쇠 비율 (0.0~1.0)
    - blend_mode: 'linear', 'exp', 'log', 'power' 중 선택

    Returns:
    - blended_audio: 감쇠 적용된 블렌딩 결과
    """
    # 위상 정렬 및 샘플 정렬
    original_audio, instrumental_audio = align_audio(original_audio, instrumental_audio)
    instrumental_audio, _ = align_signals(original_audio, instrumental_audio, max_shift=2000)

    # Blend Weight 계산
    if blend_mode == 'linear':
        w_orig = blend_ratio
    elif blend_mode == 'exp':
        w_orig = np.exp(-blend_ratio)
    elif blend_mode == 'log':
        w_orig = np.log1p(blend_ratio)
    elif blend_mode == 'power':
        w_orig = blend_ratio ** 2
    else:
        raise ValueError(f"❌ Unsupported blend_mode: {blend_mode}")

    w_inst = 1.0 - w_orig

    # Blend 처리
    blended_audio = (original_audio * w_orig) + (instrumental_audio * w_inst)

    return blended_audio


def generate_blends(original, instrumental, user_alpha, cancel_alpha, mode):

    blend_for_output = blend_audio_tracks(
        original, instrumental, blend_ratio=user_alpha, blend_mode=mode
    )
    blend_for_cancel = blend_audio_tracks(
        original, instrumental, blend_ratio=np.clip(cancel_alpha / 10.0, 0.0, 0.3), blend_mode=mode
    )
    return blend_for_output, blend_for_cancel


def residual_subtraction(original_audio, blend_audio):
    original_audio, blend_audio = align_audio(original_audio, blend_audio)
    blend_aligned, _ = align_signals(original_audio, blend_audio, max_shift=2000)
    residual_vocal = original_audio - blend_aligned
    return residual_vocal


def process_phase_cancel(original_audio, residual_vocal):
    original_audio, residual_vocal = align_audio(original_audio, residual_vocal)
    residual_vocal, _ = align_signals(original_audio, residual_vocal, max_shift=2000)
    instrumental = original_audio - residual_vocal
    return instrumental



