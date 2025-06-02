import numpy as np
import soundfile as sf
import librosa
from utils.audio_utils import match_target_amplitude
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


        target_sr = min(sr_orig, sr_blend)
        if sr_orig != target_sr:
            original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=target_sr).T
        if sr_blend != target_sr:
            blend = librosa.resample(blend.T, orig_sr=sr_blend, target_sr=target_sr).T


        if original.ndim == 1:
            original = original[:, np.newaxis]
        if blend.ndim == 1:
            blend = blend[:, np.newaxis]
        if original.shape[1] != blend.shape[1]:
            logger.warning(f"⚠️ 채널 불일치: Original({original.shape[1]}) / Blend({blend.shape[1]}) → Blend 채널 맞춤")
            blend = np.tile(blend[:, 0:1], (1, original.shape[1]))

        # Residual Vocal (보컬) 추출
        residual = original - blend
        
        # residual = match_target_amplitude(residual, original)
        max_val = np.max(np.abs(residual))
        if max_val > 0.99:
            residual = residual / (max_val * 1.01)

        sf.write(vocal_output_file, residual, target_sr)


    except Exception as e:
        logger.exception(f"❌ Residual Subtraction 처리 중 오류 발생: {e}")
        raise





def process_phase_cancel(original_file, blended_file, output_dir, base, threshold_db=-40):
    """
    Phase Cancel 방식으로 보컬 제거 처리
    - Residual Vocal: 저장 (original - blend)
    - Residual Vocal noise_gated: 저장 (Noise Gate 적용)
    - Instrumental: 반환 (저장 X)
    """
    logger = get_logger()
    try:
        residual_vocal_path = output_dir / f"{base}_vocal.wav"
        # residual_vocal_noise_gated_path = output_dir / f"{base}_vocal_noise_gated.wav"

        # === 1. Residual Vocal 생성 (Original - Blend) ===
        residual_subtraction(original_file, blended_file, str(residual_vocal_path))

        # # === 2. Noise Gate 처리 ===
        # logger.info(f"Residual Vocal에 Noise Gate 적용 중 (threshold: {threshold_db} dB)...")
        # residual_vocal, sr = sf.read(residual_vocal_path)
        # gated_vocal = noise_gate(residual_vocal, sr, threshold_db=threshold_db)
        # # norm_vocal = normalize_audio(gated_vocal)  # 필요하면 나중에 활성화
        # sf.write(str(residual_vocal_noise_gated_path), gated_vocal, sr)
        # logger.info(f"✅ Residual Vocal Noise Gated 저장 완료: {residual_vocal_noise_gated_path}")

        # === 3. Phase Cancel (Original - Residual Vocal noise_gated) ===
        original, sr_orig = sf.read(original_file)
        residual_vocal_noise_gated, sr_res = sf.read(residual_vocal_path)

        # 샘플레이트 정리
        target_sr = min(sr_orig, sr_res)
        if sr_orig != target_sr:
            original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=target_sr).T
        if sr_res != target_sr:
            residual_vocal_noise_gated = librosa.resample(residual_vocal_noise_gated.T, orig_sr=sr_res, target_sr=target_sr).T

        # 채널 정리
        if original.ndim == 1:
            original = original[:, np.newaxis]
        if residual_vocal_noise_gated.ndim == 1:
            residual_vocal_noise_gated = residual_vocal_noise_gated[:, np.newaxis]
        if original.shape[1] != residual_vocal_noise_gated.shape[1]:
            residual_vocal_noise_gated = np.tile(residual_vocal_noise_gated[:, 0:1], (1, original.shape[1]))

        # Phase Cancel 계산
        instrumental = original - residual_vocal_noise_gated
        
        # instrumental = match_target_amplitude(instrumental, original)
        max_val = np.max(np.abs(instrumental))
        if max_val > 0.99:
            instrumental /= (max_val * 1.01)

        return instrumental, target_sr

    except Exception as e:
        logger.exception(f"❌ Phase Cancel 처리 중 오류 발생: {e}")
        raise

