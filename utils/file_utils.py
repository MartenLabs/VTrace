import regex as re

def sanitize_filename(filename):
    # 유니코드 범위 내에서 허용할 문자 그룹: 한글, 일본어, 기본 ASCII
    return re.sub(r'[^\p{L}\p{N}\-_.]', '_', filename)
