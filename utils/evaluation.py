from utils.evaluate_metrics import evaluate_reconstruction
from pathlib import Path

def evaluate_results(filepath, song_output_dir, base_clean, demucs_model, logger):
    results = {}

    # Phase Cancel 결과 파일
    output_vocal = song_output_dir / f"{base_clean}_vocal_residual.wav"
    output_instrumental = song_output_dir / f"{base_clean}_instrumental_phase_cancel.wav"

    # Demucs 결과 파일
    demucs_dir = Path("separated") / demucs_model / base_clean
    demucs_vocal = demucs_dir / "vocals.wav"
    demucs_instrumental = demucs_dir / "no_vocals.wav"

    # Phase Cancel 평가 (Original vs Reconstructed)
    if filepath.exists() and output_vocal.exists() and output_instrumental.exists():
        pc_metrics = evaluate_reconstruction(
            str(filepath),
            str(output_vocal),
            str(output_instrumental),
            tag=f"{base_clean} - Phase Cancel Reconstruction"
        )
        results['Phase Cancel Reconstruction'] = pc_metrics

    # Demucs 평가 (Original vs Reconstructed)
    if filepath.exists() and demucs_vocal.exists() and demucs_instrumental.exists():
        demucs_metrics = evaluate_reconstruction(
            str(filepath),
            str(demucs_vocal),
            str(demucs_instrumental),
            tag=f"{base_clean} - Demucs Reconstruction"
        )
        results['Demucs Reconstruction'] = demucs_metrics

    # 비교 로그 출력
    if 'Phase Cancel Reconstruction' in results and 'Demucs Reconstruction' in results:
        pc = results['Phase Cancel Reconstruction']
        demucs = results['Demucs Reconstruction']
        logger.info(
            f"[비교] MSE - Phase Cancel: {pc['MSE']:.6f} vs Demucs: {demucs['MSE']:.6f} | "
            f"Cosine - Phase Cancel: {pc['Cosine']:.4f} vs Demucs: {demucs['Cosine']:.4f} | "
            f"STOI - Phase Cancel: {pc['STOI']:.4f} vs Demucs: {demucs['STOI']:.4f}"
        )

    return results
