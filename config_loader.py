# config_loader.py

import yaml
import os
from logger import get_logger

DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    logger = get_logger()

    if not os.path.exists(config_path):
        logger.warning(f"⚠️ Configuration file not found. Using default settings: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"✅ Configuration file loaded successfully: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"❌ Failed to parse configuration file: {e}")
        return {}

