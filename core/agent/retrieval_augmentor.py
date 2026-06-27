from typing import List, Dict, Any, Optional
from core.retrieve import hybrid_retriever
from utils.logger import logger
from utils.constants import RETRIEVE_TOP_K


class RetrievalAugmentor:
    def __init__(self, top_k: int = RETRIEVE_TOP_K):
        self.top_k = top_k

    def retrieve_context(
        self,
        query: str,
        business_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        results = hybrid_retriever.retrieve(
            query=query,
            business_type=business_type,
            top_k=k * 2,
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
            f"检索增强: query='{query[:50]}...' → "
            f"召回 {len(results)} 条, 去重后 {len(ranked)} 条"
        )
        return ranked

    def format_context(self, results: List[Dict[str, Any]]) -> str:
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


retrieval_augmentor = RetrievalAugmentor()
