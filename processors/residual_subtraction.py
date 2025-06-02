import numpy as np
import soundfile as sf
import librosa
from audio_utils.alignment import align_audio, align_signals
from audio_utils.loudness import peak_normalize, simple_rms_match

from logger import get_logger


def residual_subtraction(original_file, blend_file, vocal_output_file):
    """
    Residual Subtraction 방식으로 보컬 및 악기 추출
    - Vocal (Residual) = Original - Blend
    - Instrumental = Blend
    """
    logger = get_logger()

    try:
        original, sr_orig = sf.read(original_file)
        blend, sr_blend = sf.read(blend_file)

        # 샘플레이트 통일
        target_sr = min(sr_orig, sr_blend)
        if sr_orig != target_sr:
            original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=target_sr).T
        if sr_blend != target_sr:
            blend = librosa.resample(blend.T, orig_sr=sr_blend, target_sr=target_sr).T

        # 차원 통일 (모노/스테레오 처리)
        original, blend = align_audio(original, blend)

        # 위상 정렬 (cross-correlation)
        blend_aligned, lag = align_signals(original, blend, max_shift=2000)

        # Residual Vocal (보컬) 추출
        residual = original - blend_aligned

        # 정규화 (클리핑)
        # residual = simple_rms_match(residual, original)
        # residual = peak_normalize(residual)
        sf.write(vocal_output_file, residual, target_sr)

    except Exception as e:
        logger.exception(f"❌ Residual Subtraction 처리 중 오류 발생: {e}")
        raise






def process_phase_cancel(original_file, blended_file, output_dir, base):
    logger = get_logger()
    try:
        residual_vocal_path = output_dir / f"{base}_vocal.wav"

        # Residual Vocal 생성 (Original - Blend)
        residual_subtraction(original_file, blended_file, str(residual_vocal_path))

        # 파일 생성 확인
        if not residual_vocal_path.exists():
            logger.error(f"❌ residual_vocal_path 파일 생성 실패: {residual_vocal_path}")
            raise FileNotFoundError(f"Residual Vocal 파일 생성 실패: {residual_vocal_path}")

        # Residual Vocal 읽기
        original, sr_orig = sf.read(original_file)
        residual_vocal, sr_res = sf.read(residual_vocal_path)

        # 샘플레이트 통일
        target_sr = min(sr_orig, sr_res)
        if sr_orig != target_sr:
            original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=target_sr).T
        if sr_res != target_sr:
            residual_vocal = librosa.resample(residual_vocal.T, orig_sr=sr_res, target_sr=target_sr).T

        # 채널/위상 정렬
        original, residual_vocal = align_audio(original, residual_vocal)
        residual_vocal, lag = align_signals(original, residual_vocal, max_shift=2000)

        # Phase Cancel 계산
        instrumental = original - residual_vocal
        
        # instrumental = simple_rms_match(instrumental, original)
        # instrumental = peak_normalize(instrumental)
        return instrumental, target_sr

    except Exception as e:
        logger.exception(f"❌ Phase Cancel 처리 중 오류 발생: {e}")
        raise


