import regex as re
import yt_dlp
import os


def sanitize_url(url):
    return re.sub(r"\\", "", url.strip())

def youtube_download(youtube_url, output_path):
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'quiet': False,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        title = info_dict.get('title', None)
        return os.path.join(output_path, f"{title}.mp3") if title else None