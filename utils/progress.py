import re
import subprocess
from tqdm import tqdm

def run_demucs_track_as_single_bar(filepath, safe_path, demucs_model, device):
    total_models = 4
    current_model = 0
    last_progress = 0.0
    total_seconds = None
    seen_resets = 0

    with tqdm(total=1.0, desc=filepath.stem[:40], ncols=100, bar_format='{l_bar}{bar}| {n:.2%}') as pbar:
        proc = subprocess.Popen(
            [
                "demucs", "--two-stems", "vocals",
                "-n", demucs_model, "--device", device,
                str(safe_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            line = line.strip()
            # print(line)  ← ❌ 진짜 로그 숨김

            # 진행률 파싱
            progress_match = re.search(r"([\d.]+)\s*/\s*([\d.]+)", line)
            if progress_match:
                curr, total = float(progress_match.group(1)), float(progress_match.group(2))

                # 전체 길이 초기화
                if total_seconds is None:
                    total_seconds = total

                # 새 모델 시작 감지 (progress가 0 근처일 때)
                if curr < 1.0:
                    seen_resets += 1
                    current_model = min(seen_resets - 1, total_models - 1)

                model_progress = min(curr / total_seconds, 1.0)
                overall_progress = (current_model + model_progress) / total_models
                delta = overall_progress - last_progress
                if delta > 0:
                    pbar.update(delta)
                    last_progress = overall_progress

        proc.wait()
        pbar.n = 1.0
        pbar.refresh()
