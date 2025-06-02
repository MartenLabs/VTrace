import regex as re
import unicodedata
from slugify import slugify

def sanitize_filename(name, strict=False):
    name = unicodedata.normalize('NFC', name)  # 한글 깨짐 방지
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    if strict:
        name = slugify(name).replace("-", "_")
    return name.strip()

def sanitize_url(url):
    return re.sub(r"\\", "", url.strip())
