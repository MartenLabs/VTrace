import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import soundfile as sf
from processors.vtrace_core import blend_audio_tracks, residual_subtraction, process_phase_cancel
from utils.file_utils import sanitize_filename, sanitize_url
from utils.evaluation import evaluate_results
from utils.youtube import youtube_download
from audio_utils.audio_conversion import convert_wav_to_mp3
from config_loader import load_config
from logger import get_logger
import librosa

def process_file(filepath: Path, output_root: Path, alpha: float, blend_mode: str,
                 demucs_model: str, cleanup: bool, enable_eval: bool, mp3_convert: bool):
    logger = get_logger()
    base = filepath.stem
    song_output_dir = output_root / base
    song_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"🎶 {filepath.name} Demucs 분리 중...")

        result = subprocess.run(
            ["demucs", "--two-stems=vocals", "-n", demucs_model, str(filepath)]
        )

        if result.returncode != 0:
            logger.error(f"❌ Demucs 실행 오류")
            return

        sep_dir = Path("separated") / demucs_model / base
        no_vocals_path = sep_dir / "no_vocals.wav"
        if not no_vocals_path.exists():
            logger.warning(f"❌ {filepath.name} 분리 실패 (no_vocals 미생성)")
            return



        # --- 파일 읽기 ---
        original_audio, sr_orig = sf.read(str(filepath))
        instrumental_audio, sr_inst = sf.read(str(no_vocals_path))

        # 샘플레이트 통일
        sample_rate = min(sr_orig, sr_inst)
        if sr_orig != sample_rate:
            original_audio = librosa.resample(original_audio.T, orig_sr=sr_orig, target_sr=sample_rate).T
        if sr_inst != sample_rate:
            instrumental_audio = librosa.resample(instrumental_audio.T, orig_sr=sr_inst, target_sr=sample_rate).T

        # --- Blend ---
        blended_audio, sample_rate = blend_audio_tracks(
            original_audio, instrumental_audio, sample_rate,
            blend_ratio=alpha, blend_mode=blend_mode
        )

        # --- Residual Vocal ---
        residual_vocal, sample_rate = residual_subtraction(original_audio, blended_audio, sample_rate)

        # --- Phase Cancel Instrumental ---
        instrumental_phase, sample_rate = process_phase_cancel(
            original_audio, residual_vocal, sample_rate
        )

        # --- 파일 저장 ---
        base_clean = sanitize_filename(base)
        blended_output = song_output_dir / f"{base_clean}_blended.wav"
        vocal_output = song_output_dir / f"{base_clean}_vocal_residual.wav"
        instrumental_output = song_output_dir / f"{base_clean}_instrumental_phase_cancel.wav"

        sf.write(str(blended_output), blended_audio, sample_rate)
        sf.write(str(vocal_output), residual_vocal, sample_rate)
        sf.write(str(instrumental_output), instrumental_phase, sample_rate)

        logger.info(f"WAV 저장 완료: {blended_output}, {vocal_output}, {instrumental_output}")

        # --- MP3 변환 (옵션) ---
        if mp3_convert:
            for wav_path in [blended_output, vocal_output, instrumental_output]:
                mp3_path = convert_wav_to_mp3(str(wav_path))
                if mp3_path:
                    logger.info(f"MP3 변환 완료: {mp3_path}")
                else:
                    logger.warning(f"❌ MP3 변환 실패: {wav_path}")

        if enable_eval:
            evaluate_results(filepath, song_output_dir, base_clean, demucs_model, logger)

        # --- Cleanup ---
        if cleanup:
            shutil.rmtree(sep_dir.parent, ignore_errors=True)
            logger.info(f"🧹 {filepath.name} 분리 폴더 삭제 완료")

    except Exception as e:
        logger.exception(f"❌ 오류: {filepath.name} 처리 중 예외 발생: {e}")


def main():
    config = load_config()
    logger = get_logger()

    parser = argparse.ArgumentParser(description="Phase-Driven Vocal Tuner CLI")
    parser.add_argument("-i", "--input", type=str, help="입력 폴더 또는 파일 경로")
    parser.add_argument("-l", "--link", type=str, help="유튜브 링크 (MP3 다운로드 후 처리)")
    parser.add_argument("-o", "--output", type=str, help="출력 폴더 경로 (생략 시 입력 폴더 하위)")
    parser.add_argument("-a", "--alpha", type=float, help="Blend 비율 (0~1)")
    parser.add_argument("-T", "-t", "--thread", type=int, help="동시 처리할 스레드 개수")
    parser.add_argument("--blend-mode", type=str, help="Blend 방식 (linear, exp, log, power)")
    parser.add_argument("--demucs-model", type=str, help="Demucs 모델명")
    parser.add_argument("--cleanup", action="store_true", help="Demucs 분리 결과 폴더 삭제 여부")
    parser.add_argument("--convert_to_mp3", action="store_true", help="wav to mp3 convert 여부")
    parser.add_argument("--eval", action="store_true", help="MSE, Cosine, STOI 기반의 복원 평가 실행 여부")
    

    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else config.get("default_alpha", 0.1)
    blend_mode = args.blend_mode if args.blend_mode else config.get("default_blend_mode", "linear")
    demucs_model = args.demucs_model if args.demucs_model else config.get("default_demucs_model", "htdemucs_ft")
    thread_count = args.thread if args.thread is not None else config.get("max_threads", 2)
    enable_eval = args.eval or config.get("enable_evaluation", False)
    mp3_convert = args.convert_to_mp3 or config.get("enable_evaluation", False)

    files = []
    output_root = None

    if args.link:
        youtube_url = sanitize_url(args.link)
        output_dir = args.output if args.output else config.get("output_base", "downloads")
        output_root = Path(output_dir)
        logger.info(f"🎬 유튜브 링크 다운로드 중: {youtube_url}")
        mp3_path = youtube_download(youtube_url, output_path=output_root)
        if mp3_path:
            files = [Path(mp3_path)]
        else:
            logger.error("❌ 다운로드 실패")
            return
    elif args.input:
        input_path = Path(args.input)
        output_root = Path(args.output) if args.output else input_path.parent / config.get("output_base", "processed")
        exts = [f".{ext.strip().lower()}" for ext in config.get("supported_extensions", "wav,mp3,flac").split(",")]
        if input_path.is_file():
            files = [input_path] if input_path.suffix.lower() in exts else []
        elif input_path.is_dir():
            files = [f for f in input_path.glob("*") if f.suffix.lower() in exts]
        else:
            logger.error(f"❌ 입력 경로를 찾을 수 없습니다: {input_path}")
            return
    else:
        logger.error("❌ -i (파일/폴더) 또는 -l (유튜브 링크) 중 하나는 반드시 입력해야 합니다.")
        return

    if not files:
        logger.warning("⚠️ 처리할 파일이 없습니다.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎧 총 {len(files)}개 파일 처리 시작 (alpha={alpha}, blend_mode={blend_mode}, threads={thread_count}, eval={enable_eval})...")
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(process_file, f, output_root, alpha, blend_mode, demucs_model, args.cleanup, enable_eval, mp3_convert) for f in files]
        for f in as_completed(futures):
            pass

    logger.info("✅ 모든 파일 처리 완료!")

if __name__ == "__main__":
    main()
