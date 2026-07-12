from typing import List, Dict, Any, Optional
from core.vector_store.milvus_store import milvus_store
from core.es_store import es_store
from core.embedding.bge_embedding import embedding_model
from core.retrieve.graph_retriever import graph_retriever
from utils.constants import (
    RETRIEVE_TOP_K,
    SIMILARITY_THRESHOLD,
    VECTOR_WEIGHT,
    KEYWORD_WEIGHT,
    GRAPH_WEIGHT,
    ES_KEYWORD_TOP_K,
)
from utils.logger import logger


class TripleHybridRetriever:
    # 三路混合检索器：根据意图分析结果动态路由检索策略
    # - semantic: 向量(主力) + 关键词(辅助)，适合"解释/定义/说明"类问题
    # - graph: 纯图遍历，适合"投诉/购买/关联"类关系问题
    # - hybrid: 三路全开，图命中结果额外加分
    def __init__(
        self,
        top_k: int = RETRIEVE_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        vector_weight: float = VECTOR_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        graph_weight: float = GRAPH_WEIGHT,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.graph_weight = graph_weight

    def retrieve(
        self,
        query: str,
        business_type: Optional[str] = None,
        top_k: Optional[int] = None,
        retrieval_strategy: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        # 按意图分析结果选择检索策略路由
        k = top_k or self.top_k

        if retrieval_strategy == "graph":
            return self._graph_only_retrieve(query, business_type, k)
        elif retrieval_strategy == "semantic":
            return self._semantic_retrieve(query, business_type, k)
        else:
            return self._hybrid_retrieve(query, business_type, k)

    def _semantic_retrieve(
        self, query: str, business_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        # 语义检索：Milvus 向量 ANN + ES BM25 加权融合（7:3）
        expr = None
        if business_type:
            expr = f'business_type == "{business_type}"'

        query_vector = embedding_model.encode_query(query)
        vector_results = milvus_store.search(
            query_vector, top_k=top_k * 2, expr=expr
        )

        keyword_scores = self._keyword_search(query, business_type, top_k=top_k)

        fused = self._fusion_rerank(vector_results, keyword_scores, top_k, 0.7, 0.3, 0.0)
        return [r for r in fused if r["fusion_score"] >= self.similarity_threshold][:top_k]

    def _graph_only_retrieve(
        self, query: str, business_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        # 图检索：纯 Neo4j 图遍历，按排名线性降分打分
        graph_results = graph_retriever.retrieve(query, business_type, top_k)
        if graph_results:
            max_score = max(
                (r.get("fusion_score", 1.0) for r in graph_results),
                default=1.0,
            )
        else:
            max_score = 1.0

        scored = []
        for i, r in enumerate(graph_results):
            score = 1.0 - (i / max(len(graph_results), 1)) * 0.5
            scored.append({
                "id": r["id"],
                "vector_score": 0,
                "keyword_score": 0,
                "graph_score": score,
                "fusion_score": score,
                "text_chunk": r["text_chunk"],
                "source_doc": r["source_doc"],
                "business_type": r["business_type"],
            })

        return scored[:top_k]

    def _hybrid_retrieve(
        self, query: str, business_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        # 三路混合检索：向量 + 关键词 + 图，图命中结果额外加分
        expr = None
        if business_type:
            expr = f'business_type == "{business_type}"'

        query_vector = embedding_model.encode_query(query)
        vector_results = milvus_store.search(
            query_vector, top_k=top_k * 2, expr=expr
        )

        keyword_scores = self._keyword_search(query, business_type, top_k=top_k * 2)
        graph_results = graph_retriever.retrieve(query, business_type, top_k)

        fused = self._fusion_rerank(
            vector_results, keyword_scores, top_k,
            self.vector_weight, self.keyword_weight, self.graph_weight
        )

        # 对图命中的知识块进行额外加分
        graph_ids = {r["id"] for r in graph_results}
        for r in fused:
            if r["id"] in graph_ids:
                r["fusion_score"] = min(
                    1.0, r["fusion_score"] + self.graph_weight * 0.3
                )

        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        return [r for r in fused if r["fusion_score"] >= self.similarity_threshold][:top_k]

    def _keyword_search(
        self, query: str, business_type: Optional[str], top_k: int
    ) -> Dict[int, Dict[str, Any]]:
        # ES 关键词检索，返回 {文档ID: 文档元信息} 字典方便融合排序
        if not query or not query.strip():
            return {}

        es_results = es_store.search(
            query_text=query,
            top_k=top_k,
            business_type=business_type,
        )

        return {r["id"]: r for r in es_results}

    def _fusion_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_map: Dict[int, Dict[str, Any]],
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
        graph_weight: float,
    ) -> List[Dict[str, Any]]:
        # 多路结果融合排序：Min-Max 归一化 → 加权求和
        # 向量分数和关键词分数量纲不同，各归一到 [0,1] 后加权
        if not vector_results and not keyword_map:
            return []

        all_records = {}

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

        fused = []
        for record in all_records.values():
            normalized_vector = record["raw_vector_score"] / max_vector_score
            normalized_keyword = record["raw_keyword_score"] / max_keyword_score
            fusion_score = (
                vector_weight * normalized_vector
                + keyword_weight * normalized_keyword
            )
            fused.append({
                "id": record["id"],
                "vector_score": normalized_vector,
                "keyword_score": normalized_keyword,
                "graph_score": 0,
                "fusion_score": fusion_score,
                "text_chunk": record["text_chunk"],
                "source_doc": record["source_doc"],
                "business_type": record["business_type"],
            })

        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        return fused


triple_retriever = TripleHybridRetriever()
# 保留 hybrid_retriever 名称保证对外兼容
hybrid_retriever = triple_retriever
