import numpy as np
import soundfile as sf
from logger import get_logger

def match_length(reference, estimated):
    min_len = min(len(reference), len(estimated))
    return reference[:min_len], estimated[:min_len]

def calculate_sdr(reference, estimated, eps=1e-8):
    # SDR = 10 * log10 (||s_target||^2 / ||e_interf + e_artif||^2)
    error = reference - estimated
    sdr = 10 * np.log10(np.sum(reference ** 2) / (np.sum(error ** 2) + eps))
    return sdr

def calculate_sir(reference, interference, eps=1e-8):
    # SIR = 10 * log10 (||s_target||^2 / ||e_interf||^2)
    sir = 10 * np.log10(np.sum(reference ** 2) / (np.sum(interference ** 2) + eps))
    return sir

def evaluate_sdr_sir(reference_file, estimated_file, tag=""):
    logger = get_logger()
    try:
        ref, sr_ref = sf.read(reference_file)
        est, sr_est = sf.read(estimated_file)

        if ref.ndim == 1:
            ref = ref[:, np.newaxis]
        if est.ndim == 1:
            est = est[:, np.newaxis]
        if ref.shape[1] != est.shape[1]:
            logger.warning(f"⚠️ 채널 불일치: {ref.shape[1]} vs {est.shape[1]} → 채널 맞춤")
            est = np.tile(est[:, 0:1], (1, ref.shape[1]))

        if sr_ref != sr_est:
            logger.warning(f"⚠️ 샘플레이트 불일치: {sr_ref} vs {sr_est} → 최소 sr 사용")
            sr = min(sr_ref, sr_est)
        else:
            sr = sr_ref
    
        ref, est = match_length(ref, est)

        sdr = calculate_sdr(ref, est)
        sir = calculate_sir(ref, ref - est)

        logger.info(f"📊 [{tag}] SDR: {sdr:.2f} dB, SIR: {sir:.2f} dB")
        return {"SDR": sdr, "SIR": sir}

    except Exception as e:
        logger.exception(f"❌ 메트릭 계산 오류: {e}")
        return None


def calculate_dbfs(signal, eps=1e-8):
    rms = np.sqrt(np.mean(signal**2))
    dbfs = 20 * np.log10(rms + eps)
    return dbfs

def evaluate_dbfs_change(reference_file, estimated_file, tag=""):
    logger = get_logger()
    try:
        ref, sr_ref = sf.read(reference_file)
        est, sr_est = sf.read(estimated_file)

        if ref.ndim == 1:
            ref = ref[:, np.newaxis]
        if est.ndim == 1:
            est = est[:, np.newaxis]
        if ref.shape[1] != est.shape[1]:
            logger.warning(f"⚠️ 채널 불일치: {ref.shape[1]} vs {est.shape[1]} → 채널 맞춤")
            est = np.tile(est[:, 0:1], (1, ref.shape[1]))

        # dBFS 계산 (채널 평균)
        dbfs_ref = np.mean([calculate_dbfs(ref[:, c]) for c in range(ref.shape[1])])
        dbfs_est = np.mean([calculate_dbfs(est[:, c]) for c in range(est.shape[1])])
        diff = dbfs_est - dbfs_ref

        logger.info(f"📊 [dBFS] {tag} | Original: {dbfs_ref:.2f} dBFS | Estimated: {dbfs_est:.2f} dBFS | Change: {diff:.2f} dB")
        return {"Original_dBFS": dbfs_ref, "Estimated_dBFS": dbfs_est, "Change_dB": diff}

    except Exception as e:
        logger.exception(f"❌ dBFS 계산 오류: {e}")
        return None
