import yt_dlp
import os
from utils.file_utils import sanitize_filename
from logger import get_logger

def youtube_download(youtube_url, output_path):
    logger = get_logger()
    os.makedirs(output_path, exist_ok=True)

    # === 미리 safe_title 생성 ===
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            raw_title = info_dict.get('title', 'audio')
            safe_title = sanitize_filename(raw_title)
    except Exception as e:
        logger.exception(f"❌ Failed to extract metadata with yt_dlp: {e}")
        return None

    # === yt_dlp 다운로드 ===
    outtmpl = f'{output_path}/{safe_title}.%(ext)s'

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'quiet': False,
        'noplaylist': True,
        'windowsfilenames': True,  # Windows 안전성 확보
    }

    final_path = outtmpl.replace('%(ext)s', 'mp3')

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        logger.exception(f"❌ yt_dlp download failed: {e}")
        return None

    # === 파일명 검증 및 fallback 처리 ===
    if os.path.exists(final_path):
        logger.info(f"✅ Download successful: {final_path}")
        return final_path
    else:
        fallback_path = os.path.join(output_path, f"{raw_title}.mp3")
        if os.path.exists(fallback_path):
            try:
                os.rename(fallback_path, final_path)
                logger.info(f"✅ Fallback renamed: {fallback_path} -> {final_path}")
                return final_path
            except Exception as e:
                logger.exception(f"❌ Failed to rename fallback file: {e}")
                return fallback_path
        else:
            logger.error(f"❌ Downloaded file not found: {final_path} or {fallback_path}")
            return None