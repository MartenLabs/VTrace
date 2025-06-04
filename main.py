import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import subprocess
import soundfile as sf
from processors.vtrace_core import generate_blends, residual_subtraction, process_phase_cancel
from utils.file_utils import sanitize_filename, sanitize_url
from utils.evaluation import evaluate_results
from utils.youtube import youtube_download
from audio_utils.audio_conversion import convert_wav_to_mp3
from config_loader import load_config
from logger import get_logger
import librosa
import torch
import argparse

def process_file(filepath: Path, output_root: Path, blend_alpha: float, voice_alpha: float, blend_mode: str,
                 demucs_model: str, device: str, cleanup: bool, enable_eval: bool, mp3_convert: bool):
    logger = get_logger()
    base = filepath.stem
    song_output_dir = output_root / base
    song_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"🎶 Separating with Demucs: {filepath.name}")

        result = subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",
                "-n", demucs_model,
                "--device", f'{device}',
                str(filepath)
            ]
        )

        if result.returncode != 0:
            logger.error(f"❌ Demucs execution failed")
            return

        sep_dir = Path("separated") / demucs_model / base
        no_vocals_path = sep_dir / "no_vocals.wav"
        if not no_vocals_path.exists():
            logger.warning(f"❌ Separation failed for {filepath.name} (no_vocals not generated)")
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

        # --- Blend 출력용 ---
        blend_for_output, blend_for_cancel = generate_blends(
            original_audio, instrumental_audio, 
            user_alpha=blend_alpha, cancel_alpha = voice_alpha, mode=blend_mode
        )
        # --- Residual Vocal ---
        residual_vocal = residual_subtraction(
            original_audio, blend_for_cancel
        )

        # --- Phase Cancel Instrumental ---
        instrumental_phase = process_phase_cancel(
            original_audio, residual_vocal
        )

        # --- 파일 저장 ---
        base_clean = sanitize_filename(base)
        blended_output = song_output_dir / f"{base_clean}_blended.wav"
        vocal_output = song_output_dir / f"{base_clean}_vocal_residual.wav"
        instrumental_output = song_output_dir / f"{base_clean}_instrumental_phase_cancel.wav"

        sf.write(str(blended_output), blend_for_output, sample_rate)
        sf.write(str(vocal_output), residual_vocal, sample_rate)
        sf.write(str(instrumental_output), instrumental_phase, sample_rate)

        logger.info(f"WAV files saved: {blended_output}, {vocal_output}, {instrumental_output}")

        # --- MP3 변환 (옵션) ---
        if mp3_convert:
            for wav_path in [blended_output, vocal_output, instrumental_output]:
                mp3_path = convert_wav_to_mp3(str(wav_path))
                if mp3_path:
                    logger.info(f"MP3 conversion completed: {mp3_path}")
                else:
                    logger.warning(f"❌ MP3 conversion failed: {wav_path}")

        if enable_eval:
            evaluate_results(filepath, song_output_dir, base_clean, demucs_model, logger)

        # --- Cleanup ---
        if cleanup:
            shutil.rmtree(sep_dir.parent, ignore_errors=True)
            logger.info(f"🧹 Demucs folder deleted: {filepath.name}")

    except Exception as e:
        logger.exception(f"❌ Exception occurred while processing {filepath.name}: {e}")


def main():
    config = load_config()
    logger = get_logger()

    if torch.cuda.is_available():
        default_device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser = argparse.ArgumentParser(description="Phase-Driven Vocal Tuner CLI")
    parser.add_argument("-i", "--input", type=str, help="Path to input file or folder")
    parser.add_argument("-l", "--link", type=str, help="YouTube link (downloads MP3 and processes it)")
    parser.add_argument("-o", "--output", type=str, help="Path to output folder (defaults to subfolder of input)")

    parser.add_argument(
        "-ba", "--blend_alpha", type=float,
        help="Blend ratio for final output (0 to 1). Lower values attenuate vocals more and emphasize instrumentals. "
             "Extremely low values may degrade audio quality."
    )
    parser.add_argument(
        "-va", "--voice_alpha", type=float,
        help="Blend ratio for residual vocal extraction (0 to 1). Lower values emphasize vocals. "
             "Too low may cause distortion."
    )

    parser.add_argument(
        "-T", "-t", "--thread", type=int,
        help="Number of processes to run concurrently"
    )
    parser.add_argument(
        "--blend-mode", type=str, choices=["linear", "exp", "log", "power"], default="linear",
        help="Blend mode to apply ('linear', 'exp', 'log', or 'power')"
    )
    parser.add_argument(
        "--demucs-model", type=str,
        help="Name of the Demucs model to use for separation"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default=default_device,
        help="Compute device: choose from 'cpu', 'cuda', or 'mps' (default is auto-selected based on system)"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete intermediate Demucs separation results after processing"
    )
    parser.add_argument(
        "--convert_mp3", action="store_true",
        help="Convert final outputs to MP3 format"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run restoration quality evaluation (MSE, Cosine Similarity, and STOI)"
    )


    args = parser.parse_args()

    blend_alpha = args.blend_alpha if args.blend_alpha is not None else config.get("default_blend_alpha", 0.1)
    vocal_alpha = args.voice_alpha if args.voice_alpha is not None else config.get("default_vocal_alpha", 1.0)

    blend_mode = args.blend_mode if args.blend_mode else config.get("default_blend_mode", "linear")
    demucs_model = args.demucs_model if args.demucs_model else config.get("default_demucs_model", "htdemucs_ft")
    thread_count = args.thread if args.thread is not None else config.get("max_threads", 1)
    convert_mp3 = args.convert_mp3 or config.get("convert_mp3", False)
    enable_eval = args.eval or config.get("enable_evaluation", False)
    enable_cleanup = args.cleanup or config.get("cleanup", False)
    device = args.device

    files = []
    output_root = None

    if args.link:
        youtube_url = sanitize_url(args.link)
        output_dir = args.output if args.output else config.get("output_base", "downloads")
        output_root = Path(output_dir)
        logger.info(f"🎬 Downloading YouTube link: {youtube_url}")
        mp3_path = youtube_download(youtube_url, output_path=output_root)
        if mp3_path:
            files = [Path(mp3_path)]
        else:
            logger.error("❌ Download failed.")
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
            logger.error(f"❌ Input path not found: {input_path}")
            return
    else:
        logger.error("❌ Either -i (file/folder) or -l (YouTube link) must be provided.")
        return

    if not files:
        logger.warning("⚠️ No files to process.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"🎧 Starting processing of {len(files)} file(s)\n"
        f" - blend_alpha = {blend_alpha}\n"
        f" - vocal_alpha = {vocal_alpha}\n"
        f" - blend_mode = {blend_mode}\n"
        f" - device = {device}\n"
        f" - threads = {thread_count}\n"
        f" - mp3_convert = {convert_mp3}\n"
    )
    
    with ProcessPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(
                process_file, f, output_root,
                blend_alpha, vocal_alpha, blend_mode,
                demucs_model, device,
                enable_cleanup, enable_eval, convert_mp3
            )
            for f in files
        ]
        for f in as_completed(futures):
            pass
        logger.info("✅ All files processed successfully!")

if __name__ == "__main__":
    main()
