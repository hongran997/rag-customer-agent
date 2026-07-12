import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_knowledge_base")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 120
ES_HOST = os.getenv("ES_HOST", "localhost")          # ES 服务地址
ES_PORT = int(os.getenv("ES_PORT", "9200"))            # ES 服务端口
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "rag_knowledge_base")  # ES 索引名（和 Milvus collection 对应）
ES_KEYWORD_TOP_K = int(os.getenv("ES_KEYWORD_TOP_K", "50"))  # ES 单次搜索最大返回数

MIN_TEXT_LENGTH = 10

# Neo4j 图数据库配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 三路混合检索配置（向量 + 关键词 + 图）
RETRIEVE_TOP_K = 10
SIMILARITY_THRESHOLD = 0.35
VECTOR_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.25
GRAPH_WEIGHT = 0.25
GRAPH_TRAVERSAL_DEPTH = 2

COLLECTION_SCHEMA_FIELDS = [
    "id",
    "text_chunk",
    "vector",
    "source_doc",
    "business_type",
]
