from typing import List, Dict, Any, Optional
from core.graph_store.neo4j_store import neo4j_store
from utils.constants import GRAPH_TRAVERSAL_DEPTH
from utils.logger import logger


class GraphRetriever:
    # 图检索器：基于 Neo4j 知识图谱的关联检索
    # 核心思路：从用户查询中识别实体名 → 图遍历找到关联知识块
    def __init__(self, max_depth: int = GRAPH_TRAVERSAL_DEPTH):
        self.max_depth = max_depth

    def retrieve(
        self,
        query: str,
        business_type: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        # 图检索入口：匹配实体 → 查关联知识块 → 查一跳关联实体的知识块
        if not neo4j_store.is_available:
            logger.warning("Neo4j 不可用，图检索跳过")
            return []

        graph_contexts = []
        seen_chunks = set()

        q = query.strip()
        if not q:
            return []

        # 第一步：从查询中匹配已知实体名称
        entities = self._find_entities_for_query(q, business_type)
        if not entities:
            logger.info(f"图检索: 未从查询中匹配到已知实体, query='{q[:50]}'")
            return []

        for entity_name, entity_type in entities:
            # 第二步：查直接关联该实体的知识块
            related_chunks = self._find_chunks_by_entity(
                entity_name, business_type, top_k
            )
            for chunk in related_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                graph_contexts.append({
                    "id": chunk_id,
                    "text_chunk": chunk.get("text_preview", ""),
                    "source_doc": chunk.get("source_doc", ""),
                    "business_type": business_type or "default",
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                })

            # 第三步：查关联实体的邻居实体的知识块（一跳扩展）
            related_entities = self._find_related_entities(
                entity_name, business_type, max_depth=1
            )
            for rel in related_entities[:top_k]:
                related_chunks = self._find_chunks_by_entity(
                    rel["name"], business_type, top_k
                )
                for chunk in related_chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(chunk_id)
                    graph_contexts.append({
                        "id": chunk_id,
                        "text_chunk": chunk.get("text_preview", ""),
                        "source_doc": chunk.get("source_doc", ""),
                        "business_type": business_type or "default",
                        "entity_name": rel["name"],
                        "entity_type": rel.get("entity_type", ""),
                    })

            if len(graph_contexts) >= top_k:
                break

        logger.info(
            f"图检索: query='{q[:50]}' → 匹配实体 {len(entities)} 个, "
            f"关联知识块 {len(graph_contexts)} 条"
        )
        return graph_contexts[:top_k]

    def _find_entities_for_query(
        self, query: str, business_type: Optional[str]
    ) -> List[tuple]:
        # 从 Neo4j 获取当前业务域的所有实体，与查询字符串做子串匹配
        # 注意：MVP 使用简单子串匹配，后续可升级为 NLP 实体链接
        all_entities = self._get_all_entity_names(business_type)
        matched = []
        for name, etype in all_entities:
            if name in query:
                matched.append((name, etype))
        return matched

    def _get_all_entity_names(
        self, business_type: Optional[str]
    ) -> List[tuple]:
        # 查询指定业务域的所有实体名称和类型
        if business_type:
            cypher = """
            MATCH (e:Entity {business_type: $business_type})
            RETURN e.name AS name, e.entity_type AS entity_type
            """
            results = neo4j_store.run_query(cypher, {"business_type": business_type})
        else:
            cypher = """
            MATCH (e:Entity)
            RETURN e.name AS name, e.entity_type AS entity_type
            """
            results = neo4j_store.run_query(cypher)
        return [(r["name"], r["entity_type"]) for r in results]

    def _find_chunks_by_entity(
        self, entity_name: str, business_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        # 通过 MENTIONS 关系查找提及某实体的知识块
        cypher = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $entity_name})
        WHERE $business_type IS NULL OR c.business_type = $business_type
        RETURN c.chunk_id AS chunk_id, c.text_preview AS text_preview,
               c.source_doc AS source_doc
        LIMIT $top_k
        """
        return neo4j_store.run_query(cypher, {
            "entity_name": entity_name,
            "business_type": business_type,
            "top_k": top_k,
        })

    def _find_related_entities(
        self, entity_name: str, business_type: Optional[str], max_depth: int
    ) -> List[Dict[str, Any]]:
        # 图遍历：查找与目标实体有直接或间接关联的其他实体
        # 用于多跳关系场景，如 "客户A投诉了哪些产品" 需要 客户→投诉→产品 的路径
        cypher = f"""
        MATCH (e:Entity {{name: $entity_name}})-[r*1..{max_depth}]-(related:Entity)
        WHERE $business_type IS NULL OR related.business_type = $business_type
        RETURN DISTINCT related.name AS name, related.entity_type AS entity_type
        LIMIT 20
        """
        return neo4j_store.run_query(cypher, {
            "entity_name": entity_name,
            "business_type": business_type,
        })

    def format_graph_context(
        self, results: List[Dict[str, Any]]
    ) -> str:
        # 将图检索结果格式化为供 LLM 使用的上下文字符串
        if not results:
            return ""
        sections = []
        for i, r in enumerate(results):
            entity_info = f" (关联实体: {r.get('entity_name', '')})" if r.get("entity_name") else ""
            section = (
                f"[图关联知识 {i + 1}]"
                f"(来源: {r.get('source_doc', '未知')}"
                f"{entity_info})\n"
                f"{r['text_chunk']}"
            )
            sections.append(section)
        return "\n\n".join(sections)


graph_retriever = GraphRetriever()
