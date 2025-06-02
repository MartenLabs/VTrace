# config_loader.py

import yaml
import os
from logger import get_logger

DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    logger = get_logger()

    if not os.path.exists(config_path):
        logger.warning(f"⚠️ 설정 파일이 없습니다. 기본값으로 진행: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"✅ 설정 파일 로드 완료: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"❌ 설정 파일 파싱 오류: {e}")
        return {}

