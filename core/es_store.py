from typing import List, Optional, Dict, Any
from elasticsearch import Elasticsearch, helpers
from utils.logger import logger
from utils.constants import ES_HOST, ES_PORT, ES_INDEX_NAME


class ESStore:
    # 单例模式，整个应用共用一个 ES 客户端
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: str = ES_HOST,
        port: int = ES_PORT,
        index_name: str = ES_INDEX_NAME,
    ):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.host = host
        self.port = port
        self.index_name = index_name
        # 连接 ES 服务，默认 http://localhost:9200
        self.client = Elasticsearch(f"http://{host}:{port}")
        # 索引不存在则自动创建
        self._create_index_if_not_exists()
        logger.info(
            f"ES 连接成功: {host}:{port}, 索引: {index_name}"
        )

    def _create_index_if_not_exists(self):
        # 索引已存在则跳过
        if self.client.indices.exists(index=self.index_name):
            return
        # 定义索引的字段映射（对标 Milvus collection schema）
        # text_chunk 用 text 类型 + standard 分词器，支持全文检索
        # 其他字段用 keyword 类型，只做精确匹配过滤
        mapping = {
            "mappings": {
                "properties": {
                    "text_chunk": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "source_doc": {
                        "type": "keyword"
                    },
                    "business_type": {
                        "type": "keyword"
                    },
                    "chunk_index": {
                        "type": "integer"
                    }
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"创建 ES 索引: {self.index_name}")

    def insert(
        self,
        ids: List[int],
        texts: List[str],
        source_docs: Optional[List[str]] = None,
        business_types: Optional[List[str]] = None,
        chunk_indices: Optional[List[int]] = None,
    ):
        # 用 Milvus 返回的主键 ID 作为 ES 文档 _id，保证两个库数据一一对应
        if source_docs is None:
            source_docs = [""] * len(texts)
        if business_types is None:
            business_types = [""] * len(texts)
        if chunk_indices is None:
            chunk_indices = list(range(len(texts)))

        # 组装批量写入的 action 列表
        actions = []
        for i, doc_id in enumerate(ids):
            doc = {
                "text_chunk": texts[i],
                "source_doc": source_docs[i],
                "business_type": business_types[i],
                "chunk_index": chunk_indices[i],
            }
            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": doc,
            })

        # helpers.bulk 批量写入，不因单条失败中断
        success, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        if errors:
            logger.error(f"ES 批量写入部分失败: {errors[:3]}")
        logger.info(f"ES 批量写入完成: {success} 条")

    def search(
        self,
        query_text: str,
        top_k: int = 50,
        business_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # ES 全文检索：用 match 做中文分词匹配，支持传入业务类型过滤
        must_queries = [{"match": {"text_chunk": query_text}}]
        if business_type:
            must_queries.append({"term": {"business_type": business_type}})

        body = {
            "query": {
                "bool": {
                    "must": must_queries
                }
            },
            "size": top_k,
        }

        response = self.client.search(index=self.index_name, body=body)
        hits = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            # 返回格式和 Milvus search 保持一致，方便混合检索器融合
            hits.append({
                "id": int(hit["_id"]),
                "score": hit["_score"],
                "text_chunk": source["text_chunk"],
                "source_doc": source.get("source_doc", ""),
                "business_type": source.get("business_type", ""),
            })
        return hits

    def delete_by_business_type(self, business_type: str):
        # 按业务类型批量删除（配合 Milvus 的数据清理）
        body = {
            "query": {
                "term": {"business_type": business_type}
            }
        }
        result = self.client.delete_by_query(
            index=self.index_name, body=body, refresh=True
        )
        logger.info(
            f"ES 删除业务类型数据: {business_type}, "
            f"删除 {result['deleted']} 条"
        )

    def count(self) -> int:
        response = self.client.count(index=self.index_name)
        return response["count"]


# 全局单例，和其他 store 保持一致的导入方式
es_store = ESStore()
