import logging
import os
from datetime import datetime
import yaml

_logger_instance = None

def get_logger(config_path="config.yaml"):
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    # === config.yaml 로딩 ===
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_level_str = config.get("log_level", "INFO").upper()
    log_dir = config.get("log_dir", "logs")
    log_level = getattr(logging, log_level_str, logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"vtrace_{timestamp}.log")

    logger = logging.getLogger("VTrace")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.propagate = False

    logger.info(f"Logger initialized (Level: {log_level_str}, File: {log_file})")

    _logger_instance = logger
    return _logger_instance
