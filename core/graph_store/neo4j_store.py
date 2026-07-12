from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from utils.constants import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.logger import logger


class Neo4jStore:
    # 单例模式：全局共享一个 Neo4j 连接池
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        self._connect()
        # 启动时自动创建唯一约束，确保数据一致性
        self._ensure_constraints()

    def _connect(self):
        # 建立 Bolt 连接并验证连通性，失败时降级但不抛异常
        try:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self._driver.verify_connectivity()
            logger.info(f"Neo4j 连接成功: {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j 连接失败 ({self.uri}): {e}，图检索功能不可用")
            self._driver = None

    def _ensure_constraints(self):
        # 创建唯一约束：确保 Chunk、Document、Entity 不重复
        if not self._driver:
            return
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.source_doc IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.business_type) IS UNIQUE",
        ]
        with self._driver.session() as session:
            for cypher in constraints:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning(f"Neo4j 约束创建失败: {e}")

    @property
    def is_available(self) -> bool:
        return self._driver is not None

    def _get_session(self) -> Optional[Session]:
        if not self._driver:
            return None
        return self._driver.session()

    def run_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # 通用 Cypher 查询接口，自动管理 session 生命周期
        session = self._get_session()
        if not session:
            return []
        try:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j 查询失败: {e}\nCypher: {cypher}")
            return []
        finally:
            session.close()

    def merge_chunk(
        self,
        chunk_id: int,
        text_preview: str,
        source_doc: str,
        business_type: str,
        chunk_index: int,
    ):
        # 创建或更新知识块节点（与 Milvus/ES 的文档块一一对应）
        if not self._driver:
            return
        cypher = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text_preview = $text_preview,
            c.source_doc = $source_doc,
            c.business_type = $business_type,
            c.chunk_index = $chunk_index
        """
        self.run_query(cypher, {
            "chunk_id": chunk_id,
            "text_preview": text_preview[:200],
            "source_doc": source_doc,
            "business_type": business_type,
            "chunk_index": chunk_index,
        })

    def merge_document(self, source_doc: str, business_type: str):
        # 创建或更新文档节点，作为知识块的归属容器
        if not self._driver:
            return
        cypher = """
        MERGE (d:Document {source_doc: $source_doc})
        SET d.business_type = $business_type
        """
        self.run_query(cypher, {
            "source_doc": source_doc,
            "business_type": business_type,
        })

    def link_chunk_to_document(self, chunk_id: int, source_doc: str):
        # 建立知识块 → 文档的从属关系
        if not self._driver:
            return
        cypher = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MATCH (d:Document {source_doc: $source_doc})
        MERGE (c)-[:CHUNK_OF]->(d)
        """
        self.run_query(cypher, {
            "chunk_id": chunk_id,
            "source_doc": source_doc,
        })

    def upsert_entity(
        self, name: str, entity_type: str, business_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        # 创建或更新知识图谱实体节点（如客户、产品、人员等）
        if not self._driver:
            return
        params = {
            "name": name,
            "entity_type": entity_type,
            "business_type": business_type,
        }
        if properties:
            set_clauses = ", ".join(
                f"e.{k} = ${k}" for k in properties
            )
            cypher = f"""
            MERGE (e:Entity {{name: $name, business_type: $business_type}})
            SET e.entity_type = $entity_type, {set_clauses}
            """
            params.update(properties)
        else:
            cypher = """
            MERGE (e:Entity {name: $name, business_type: $business_type})
            SET e.entity_type = $entity_type
            """
        self.run_query(cypher, params)

    def link_entity_to_chunk(self, entity_name: str, chunk_id: int, business_type: str):
        # 建立知识块 → 实体的提及关系（Chunk 中提到某实体）
        if not self._driver:
            return
        cypher = """
        MATCH (e:Entity {name: $entity_name, business_type: $business_type})
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MERGE (c)-[:MENTIONS]->(e)
        """
        self.run_query(cypher, {
            "entity_name": entity_name,
            "chunk_id": chunk_id,
            "business_type": business_type,
        })

    def create_relationship(
        self,
        from_name: str,
        rel_type: str,
        to_name: str,
        business_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        # 在实体之间建立业务关系，如 客户→投诉→产品、人员→负责→客户
        if not self._driver:
            return
        props_str = ""
        params = {
            "from_name": from_name,
            "to_name": to_name,
            "business_type": business_type,
        }
        if properties:
            set_items = []
            for k, v in properties.items():
                param_key = f"prop_{k}"
                set_items.append(f"r.{k} = ${param_key}")
                params[param_key] = v
            if set_items:
                props_str = "SET " + ", ".join(set_items)

        cypher = f"""
        MATCH (a:Entity {{name: $from_name, business_type: $business_type}})
        MATCH (b:Entity {{name: $to_name, business_type: $business_type}})
        MERGE (a)-[r:{rel_type}]->(b)
        {props_str}
        """
        self.run_query(cypher, params)

    def delete_by_business_type(self, business_type: str):
        # 按业务类型级联删除所有相关节点和关系
        if not self._driver:
            return
        cypher = """
        MATCH (n {business_type: $business_type})
        DETACH DELETE n
        """
        self.run_query(cypher, {"business_type": business_type})

    def get_entity_by_name(
        self, name: str, business_type: str
    ) -> Optional[Dict[str, Any]]:
        # 按名称查找实体节点，用于检索时的实体匹配
        cypher = """
        MATCH (e:Entity {name: $name, business_type: $business_type})
        RETURN e.name AS name, e.entity_type AS entity_type, e.business_type AS business_type
        """
        results = self.run_query(cypher, {
            "name": name, "business_type": business_type
        })
        return results[0] if results else None

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j 连接已关闭")

    def __del__(self):
        self.close()


neo4j_store = Neo4jStore()
