import numpy as np
import soundfile as sf
from pystoi import stoi
from logger import get_logger
import librosa

def match_length(reference, estimated):
    min_len = min(reference.shape[0], estimated.shape[0])
    return reference[:min_len], estimated[:min_len]

def calculate_mse(reference, estimated):
    error = reference - estimated
    return np.mean(error ** 2)

def calculate_cosine_similarity(reference, estimated):
    ref_norm = reference / (np.linalg.norm(reference) + 1e-8)
    est_norm = estimated / (np.linalg.norm(estimated) + 1e-8)
    return np.sum(ref_norm * est_norm)

def evaluate_reconstruction(original_file, vocal_file, instrumental_file, tag=""):
    logger = get_logger()
    try:
        # --- Load audio ---
        original, sr_orig = sf.read(original_file)
        vocal, sr_vocal = sf.read(vocal_file)
        instrumental, sr_inst = sf.read(instrumental_file)

        # --- Channel handling ---
        for var_name, audio in zip(["original", "vocal", "instrumental"], [original, vocal, instrumental]):
            if audio.ndim == 1:
                logger.warning(f"⚠️ {var_name} is mono, converting to (N,1)")
                audio = audio[:, np.newaxis]

        min_sr = min(sr_orig, sr_vocal, sr_inst)
        if sr_orig != min_sr:
            original = librosa.resample(original.T, orig_sr=sr_orig, target_sr=min_sr).T
        if sr_vocal != min_sr:
            vocal = librosa.resample(vocal.T, orig_sr=sr_vocal, target_sr=min_sr).T
        if sr_inst != min_sr:
            instrumental = librosa.resample(instrumental.T, orig_sr=sr_inst, target_sr=min_sr).T

        # --- Length match ---
        original, vocal = match_length(original, vocal)
        original, instrumental = match_length(original, instrumental)

        # --- Reconstruct ---
        reconstructed = vocal + instrumental
        original, reconstructed = match_length(original, reconstructed)

        # --- MSE, Cosine (채널별 평균) ---
        mse = np.mean([calculate_mse(original[:, c], reconstructed[:, c]) for c in range(original.shape[1])])
        cosine_sim = np.mean([calculate_cosine_similarity(original[:, c], reconstructed[:, c]) for c in range(original.shape[1])])

        # --- STOI (모노 평균) ---
        original_mono = np.mean(original, axis=1) if original.ndim == 2 else original
        reconstructed_mono = np.mean(reconstructed, axis=1) if reconstructed.ndim == 2 else reconstructed
        original_mono = np.squeeze(original_mono)
        reconstructed_mono = np.squeeze(reconstructed_mono)

        try:
            stoi_score = stoi(original_mono, reconstructed_mono, min_sr, extended=False)
        except Exception as e:
            logger.warning(f"⚠️ STOI 계산 실패: {e}")
            stoi_score = None

        # --- 로그 출력 ---
        logger.info(f"[{tag}] MSE: {mse:.6f}, Cosine: {cosine_sim:.4f}, STOI: {stoi_score if stoi_score is not None else 'N/A'}")

        return {
            "MSE": mse,
            "Cosine": cosine_sim,
            "STOI": stoi_score
        }

    except Exception as e:
        logger.exception(f"❌ 평가 메트릭 계산 오류: {e}")
        return None
