from typing import List, Optional, Dict, Any
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
from utils.logger import logger
from utils.constants import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_DIM,
    RETRIEVE_TOP_K,
)


class MilvusStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        collection_name: str = MILVUS_COLLECTION_NAME,
        dim: int = EMBEDDING_DIM,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._connect()
        self.collection = self._get_or_create_collection()
        self.collection.load()
        logger.info(
            f"Milvus 连接成功: {host}:{port}, "
            f"集合: {collection_name}, 向量维度: {dim}"
        )

    def _connect(self):
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="business_type", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="RAG 客服知识库")
        collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index("vector", index_params)
        logger.info(f"创建 Milvus 集合: {self.collection_name}")
        return collection

    def insert(
        self,
        texts: List[str],
        vectors: np.ndarray,
        source_docs: Optional[List[str]] = None,
        business_types: Optional[List[str]] = None,
        chunk_indices: Optional[List[int]] = None,
    ):
        if source_docs is None:
            source_docs = [""] * len(texts)
        if business_types is None:
            business_types = [""] * len(texts)
        if chunk_indices is None:
            chunk_indices = list(range(len(texts)))

        entities = [
            texts,
            vectors.tolist(),
            source_docs,
            business_types,
            chunk_indices,
        ]
        mr = self.collection.insert(entities)
        self.collection.flush()
        logger.info(
            f"向量入库完成: {len(texts)} 条, "
            f"主键范围: {mr.primary_keys[:3] if mr.primary_keys else 'N/A'}"
        )
        return mr

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = RETRIEVE_TOP_K,
        expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.collection.load()
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16},
        }
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text_chunk", "source_doc", "business_type"],
        )
        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "text_chunk": hit.entity.get("text_chunk"),
                "source_doc": hit.entity.get("source_doc"),
                "business_type": hit.entity.get("business_type"),
            })
        return hits

    def delete_by_business_type(self, business_type: str):
        expr = f'business_type == "{business_type}"'
        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"删除业务类型数据: {business_type}")

    def count(self, expr: Optional[str] = None) -> int:
        self.collection.load()
        return self.collection.query(expr=expr, output_fields=["count(*)"])[0]["count(*)"]


milvus_store = MilvusStore()
