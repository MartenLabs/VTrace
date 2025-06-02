import numpy as np
import subprocess
import os


def match_target_loudness(target_signal, reference_signal):
    """
    target_signal의 평균 dBFS를 reference_signal의 평균 dBFS에 맞춤
    """
    def rms_db(signal):
        rms = np.sqrt(np.mean(signal**2)) + 1e-8
        db = 20 * np.log10(rms)
        return db

    ref_db = rms_db(reference_signal)
    target_db = rms_db(target_signal)

    gain_db = ref_db - target_db
    gain_linear = 10 ** (gain_db / 20)

    return target_signal * gain_linear


def convert_wav_to_mp3(input_wav, output_mp3=None, sample_rate=44100, quality=0):
    """
    WAV 파일을 MP3로 변환 (FFmpeg 사용)
    
    Args:
        input_wav (str): 입력 WAV 파일 경로
        output_mp3 (str): 출력 MP3 파일 경로 (없으면 자동 생성)
        sample_rate (int): 샘플레이트 (default: 44100)
        quality (int): 음질 (0=최고, 9=최저) (default: 0)
    """
    if not output_mp3:
        base, _ = os.path.splitext(input_wav)
        output_mp3 = f"{base}.mp3"

    command = [
        "ffmpeg", "-y",  # 덮어쓰기 허용
        "-i", input_wav,
        "-codec:a", "libmp3lame",
        "-qscale:a", str(quality),
        "-ar", str(sample_rate),
        output_mp3
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ MP3 변환 완료: {output_mp3}")
        return output_mp3
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 변환 실패: {e.stderr.decode()}")
        return None
