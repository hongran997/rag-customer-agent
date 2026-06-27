from typing import List, Dict, Any, Optional
import numpy as np
import jieba
from core.vector_store.milvus_store import milvus_store
from core.embedding.bge_embedding import embedding_model
from utils.constants import (
    RETRIEVE_TOP_K,
    SIMILARITY_THRESHOLD,
    VECTOR_WEIGHT,
    KEYWORD_WEIGHT,
)
from utils.logger import logger


class HybridRetriever:
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
        k = top_k or self.top_k
        expr = None
        if business_type:
            expr = f'business_type == "{business_type}"'

        query_vector = embedding_model.encode_query(query)
        vector_results = milvus_store.search(
            query_vector, top_k=k * 2, expr=expr
        )

        if not vector_results:
            return []

        keyword_results = self._keyword_search(query, vector_results)

        fused_results = self._fusion_rerank(
            vector_results, keyword_results, k
        )

        filtered = [
            r for r in fused_results
            if r["fusion_score"] >= self.similarity_threshold
        ]

        return filtered[:k]

    def _keyword_search(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        keywords = set(jieba.lcut(query))
        keyword_scores = {}

        for candidate in candidates:
            text = candidate["text_chunk"]
            hits = sum(1 for kw in keywords if kw in text)
            score = hits / max(len(keywords), 1)
            keyword_scores[candidate["id"]] = score

        return keyword_scores

    def _fusion_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_scores: Dict[int, float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not vector_results:
            return []

        max_vector_score = max(r["score"] for r in vector_results) or 1.0
        max_keyword_score = max(keyword_scores.values()) or 1.0

        for r in vector_results:
            normalized_vector = r["score"] / max_vector_score
            normalized_keyword = keyword_scores.get(r["id"], 0) / max_keyword_score
            r["vector_score"] = normalized_vector
            r["keyword_score"] = normalized_keyword
            r["fusion_score"] = (
                self.vector_weight * normalized_vector
                + self.keyword_weight * normalized_keyword
            )

        vector_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        return vector_results


hybrid_retriever = HybridRetriever()
