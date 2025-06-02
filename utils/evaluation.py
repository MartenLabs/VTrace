from utils.evaluate_metrics import evaluate_sdr_sir, evaluate_dbfs_change
from pathlib import Path

def evaluate_results(filepath, song_output_dir, base_clean, demucs_model, logger):
    results = {}

    output_vocal = song_output_dir / f"{base_clean}_vocal_normalized.wav"
    output_instrumental = song_output_dir / f"{base_clean}_instrumental_phase_cancel_norm.wav"

    demucs_dir = Path("separated") / demucs_model / base_clean
    demucs_vocal = demucs_dir / "vocals.wav"
    demucs_instrumental = demucs_dir / "no_vocals.wav"

    if filepath.exists() and output_instrumental.exists():
        pc_inst = evaluate_sdr_sir(str(filepath), str(output_instrumental), tag=f"{base_clean} - Instrumental vs Original")
        results['Phase Cancel Instrumental'] = pc_inst

    if filepath.exists() and output_vocal.exists():
        pc_vocal = evaluate_sdr_sir(str(filepath), str(output_vocal), tag=f"{base_clean} - Vocal vs Original")
        results['Phase Cancel Vocal'] = pc_vocal

    if filepath.exists() and output_vocal.exists():
        evaluate_dbfs_change(str(filepath), str(output_vocal), tag=f"{base_clean} - Vocal dBFS Change")

    if filepath.exists() and demucs_instrumental.exists():
        demucs_inst = evaluate_sdr_sir(str(filepath), str(demucs_instrumental), tag=f"{base_clean} - Demucs Instrumental vs Original")
        results['Demucs Instrumental'] = demucs_inst

    if filepath.exists() and demucs_vocal.exists():
        demucs_vocal_res = evaluate_sdr_sir(str(filepath), str(demucs_vocal), tag=f"{base_clean} - Demucs Vocal vs Original")
        results['Demucs Vocal'] = demucs_vocal_res

    if all(k in results for k in ['Phase Cancel Instrumental', 'Demucs Instrumental']):
        logger.info(f"🔍 [비교] Phase Cancel Instrumental SDR: {results['Phase Cancel Instrumental']['SDR']:.2f} dB vs Demucs Instrumental SDR: {results['Demucs Instrumental']['SDR']:.2f} dB")

    if all(k in results for k in ['Phase Cancel Vocal', 'Demucs Vocal']):
        logger.info(f"🔍 [비교] Phase Cancel Vocal SDR: {results['Phase Cancel Vocal']['SDR']:.2f} dB vs Demucs Vocal SDR: {results['Demucs Vocal']['SDR']:.2f} dB")
