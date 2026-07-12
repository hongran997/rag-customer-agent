from typing import List, Dict, Any, Optional
from core.retrieve import hybrid_retriever
from core.retrieve.graph_retriever import graph_retriever
from utils.logger import logger
from utils.constants import RETRIEVE_TOP_K


class RetrievalAugmentor:
    # 检索增强模块：对外提供统一的检索和上下文格式化接口
    # 内部路由到 TripleHybridRetriever，support 多策略
    def __init__(self, top_k: int = RETRIEVE_TOP_K):
        self.top_k = top_k

    def retrieve_context(
        self,
        query: str,
        business_type: Optional[str] = None,
        retrieval_strategy: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # 按指定策略检索并去重
        k = top_k or self.top_k
        results = hybrid_retriever.retrieve(
            query=query,
            business_type=business_type,
            top_k=k * 2,
            retrieval_strategy=retrieval_strategy,
        )

        seen_texts = set()
        deduplicated = []
        for r in results:
            text = r["text_chunk"]
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(r)

        ranked = deduplicated[:k]
        logger.info(
            f"检索增强: query='{query[:50]}...' "
            f"[策略={retrieval_strategy}] → "
            f"召回 {len(results)} 条, 去重后 {len(ranked)} 条"
        )
        return ranked

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        # 将检索结果格式化为给 LLM 的上下文（含来源和相关性标记）
        if not results:
            return ""

        sections = []
        for i, r in enumerate(results):
            section = (
                f"[知识片段 {i + 1}] "
                f"(来源: {r.get('source_doc', '未知')}, "
                f"相关性: {r.get('fusion_score', 0):.3f})\n"
                f"{r['text_chunk']}"
            )
            sections.append(section)

        return "\n\n".join(sections)

    def build_rich_context(
        self,
        semantic_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> str:
        # 构建富上下文：同时展示语义检索和图检索的结果，分区域隔开
        parts = []
        if semantic_results:
            parts.append("【语义检索结果】\n" + self.format_context(semantic_results))
        if graph_results:
            parts.append("【图关联检索结果】\n" + graph_retriever.format_graph_context(graph_results))
        return "\n\n".join(parts)


retrieval_augmentor = RetrievalAugmentor()
