from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from utils.logger import logger
from utils import constants


DEFAULT_CONFIG_PATH = Path(__file__).parent / "business_config_template.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
        return get_default_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载业务配置: {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        return load_config(str(config_path))
    return {}


def apply_config(config: Dict[str, Any]):
    business = config.get("business", {})
    retriever_cfg = config.get("retrieval", {})
    chunker_cfg = config.get("chunker", {})
    llm_cfg = config.get("llm", {})
    session_cfg = config.get("session", {})

    constants.RETRIEVE_TOP_K = retriever_cfg.get("top_k", constants.RETRIEVE_TOP_K)
    constants.SIMILARITY_THRESHOLD = retriever_cfg.get(
        "similarity_threshold", constants.SIMILARITY_THRESHOLD
    )
    constants.VECTOR_WEIGHT = retriever_cfg.get(
        "vector_weight", constants.VECTOR_WEIGHT
    )
    constants.KEYWORD_WEIGHT = retriever_cfg.get(
        "keyword_weight", constants.KEYWORD_WEIGHT
    )
    constants.CHUNK_SIZE = chunker_cfg.get("chunk_size", constants.CHUNK_SIZE)
    constants.CHUNK_OVERLAP = chunker_cfg.get(
        "chunk_overlap", constants.CHUNK_OVERLAP
    )
    constants.SESSION_TTL = session_cfg.get("ttl", constants.SESSION_TTL)

    logger.info(
        f"已应用业务配置: {business.get('name', 'default')}"
    )

    return business.get("name", "default")
