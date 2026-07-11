from typing import List, Dict, Any, Optional
from core.vector_store.milvus_store import milvus_store
from core.es_store import es_store
from core.embedding.bge_embedding import embedding_model
from utils.constants import (
    RETRIEVE_TOP_K,
    SIMILARITY_THRESHOLD,
    VECTOR_WEIGHT,
    KEYWORD_WEIGHT,
    ES_KEYWORD_TOP_K,
)
from utils.logger import logger


class HybridRetriever:
    # 混合检索器：向量检索 + ES 全文检索 加权融合
    def __init__(
        self,
        top_k: int = RETRIEVE_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        vector_weight: float = VECTOR_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        business_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # 两路召回：Milvus 向量检索 + ES 全文检索
        # 各自独立召回后做加权融合排序
        k = top_k or self.top_k
        expr = None
        if business_type:
            expr = f'business_type == "{business_type}"'

        # 第一路：向量语义检索（Milvus ANN）
        query_vector = embedding_model.encode_query(query)

        vector_results = milvus_store.search(
            query_vector, top_k=k * 2, expr=expr
        )

        # 第二路：ES 全文检索（倒排索引 BM25）
        keyword_scores = self._keyword_search(query, business_type, top_k=k * 2)

        # 两路结果合并去重 + 加权融合
        fused_results = self._fusion_rerank(
            vector_results, keyword_scores, k
        )

        # 低于相似度阈值的过滤掉
        filtered = [
            r for r in fused_results
            if r["fusion_score"] >= self.similarity_threshold
        ]

        return filtered[:k]

    def _keyword_search(
        self, query: str, business_type: Optional[str], top_k: int
    ) -> Dict[int, float]:
        # 调 ES 做全库关键词搜索，返回 {doc_id: 文档详情} 的字典
        if not query or not query.strip():
            return {}

        es_results = es_store.search(
            query_text=query,
            top_k=top_k,
            business_type=business_type,
        )

        # 转成 dict，方便 fusion_rerank 按 id 快速查找
        return {r["id"]: r for r in es_results}

    def _fusion_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_map: Dict[int, Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        # 合并向量检索和 ES 检索的结果
        # 两路分数的量纲不同（向量是 IP 内积，ES 是 BM25），先各自归一化到 [0,1]
        # 再按配置权重加权：fusion_score = W_v * norm_vector + W_k * norm_keyword
        if not vector_results and not keyword_map:
            return []

        # 用 dict 做合并去重，key 是文档 id
        all_records = {}

        # 先把向量结果放进去
        max_vector_score = 1.0
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results) or 1.0
            for r in vector_results:
                doc_id = r["id"]
                all_records[doc_id] = {
                    "id": doc_id,
                    "raw_vector_score": r["score"],
                    "text_chunk": r["text_chunk"],
                    "source_doc": r["source_doc"],
                    "business_type": r["business_type"],
                    "raw_keyword_score": 0,
                }

        # 再把 ES 结果合并进去（重复的补上 keyword_score，不重复的加进来）
        max_keyword_score = 1.0
        if keyword_map:
            max_keyword_score = max(
                v["score"] for v in keyword_map.values()
            ) or 1.0
            for doc_id, es_info in keyword_map.items():
                if doc_id in all_records:
                    all_records[doc_id]["raw_keyword_score"] = es_info["score"]
                else:
                    all_records[doc_id] = {
                        "id": doc_id,
                        "raw_vector_score": 0,
                        "text_chunk": es_info["text_chunk"],
                        "source_doc": es_info["source_doc"],
                        "business_type": es_info["business_type"],
                        "raw_keyword_score": es_info["score"],
                    }

        # 归一化 + 加权融合
        fused = []
        for record in all_records.values():
            normalized_vector = record["raw_vector_score"] / max_vector_score
            normalized_keyword = record["raw_keyword_score"] / max_keyword_score
            fusion_score = (
                self.vector_weight * normalized_vector
                + self.keyword_weight * normalized_keyword
            )
            fused.append({
                "id": record["id"],
                "vector_score": normalized_vector,
                "keyword_score": normalized_keyword,
                "fusion_score": fusion_score,
                "text_chunk": record["text_chunk"],
                "source_doc": record["source_doc"],
                "business_type": record["business_type"],
            })

        # 按融合分降序排列
        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        return fused


# 全局单例
hybrid_retriever = HybridRetriever()
