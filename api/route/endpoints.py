from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from api.route.models import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    BatchUploadResponse,
    SessionHistoryRequest,
    ErrorResponse,
    GraphEntityRequest,
    GraphRelationshipRequest,
    GraphQueryRequest,
)
from core.agent.rag_agent import rag_agent
from core.pipeline import process_single_document, process_folder
from api.session_store import session_manager
from api.stream.stream_helper import stream_response
from core.graph_store.neo4j_store import neo4j_store
from core.retrieve.graph_retriever import graph_retriever
from utils.logger import logger
import tempfile
import os
from pathlib import Path

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="问答对话",
    description="支持单轮/多轮问答，根据 stream 参数决定是否流式返回",
)
async def chat(request: ChatRequest):
    try:
        if request.stream:
            return stream_response(
                rag_agent.stream_run(
                    query=request.query,
                    session_id=request.session_id,
                    business_type=request.business_type,
                )
            )

        result = rag_agent.run(
            query=request.query,
            session_id=request.session_id,
            business_type=request.business_type,
        )
        return ChatResponse(
            answer=result.get("answer", ""),
            answer_with_source=result.get("answer_with_source", ""),
            session_id=request.session_id,
            intent_type=result.get("intent_type", "business_consult"),
            reference_source=result.get("reference_source", []),
            is_hallucination=result.get("is_hallucination", False),
        )
    except Exception as e:
        logger.error(f"问答接口异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    business_type: str = Form(default="default"),
):
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".md", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {suffix}",
            )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        chunks = process_single_document(tmp_path, business_type)
        os.unlink(tmp_path)

        return DocumentUploadResponse(
            file=file.filename,
            chunks=len(chunks),
            status="success",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文档异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/knowledge/batch-upload",
    response_model=BatchUploadResponse,
)
async def batch_upload_documents(
    business_type: str = Form(default="default"),
    files: List[UploadFile] = File(...),
):
    try:
        batch_dir = tempfile.mkdtemp()
        saved_files = []

        for file in files:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".pdf", ".docx", ".md", ".txt"}:
                continue
            tmp_path = os.path.join(batch_dir, file.filename)
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            saved_files.append(tmp_path)

        stats = process_folder(batch_dir, business_type)

        for f in saved_files:
            os.unlink(f)
        os.rmdir(batch_dir)

        return BatchUploadResponse(
            total_files=stats["total_files"],
            total_chunks=stats["total_chunks"],
            files=stats["files"],
        )
    except Exception as e:
        logger.error(f"批量上传异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge/{business_type}")
async def delete_knowledge(business_type: str):
    # 删除知识库时同步清理三个存储系统：Milvus（向量）、ES（全文索引）、Neo4j（图谱）
    try:
        from core.vector_store import milvus_store
        from core.es_store import es_store
        milvus_store.delete_by_business_type(business_type)
        es_store.delete_by_business_type(business_type)
        neo4j_store.delete_by_business_type(business_type)
        return {"status": "success", "business_type": business_type}
    except Exception as e:
        logger.error(f"删除知识库异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/count")
async def count_knowledge(business_type: Optional[str] = None):
    try:
        from core.vector_store import milvus_store
        expr = (
            f'business_type == "{business_type}"'
            if business_type else None
        )
        count = milvus_store.count(expr)
        return {"count": count, "business_type": business_type or "all"}
    except Exception as e:
        logger.error(f"统计知识库异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session_history(session_id: str, max_turns: int = 10):
    try:
        history = session_manager.get_history(session_id, max_turns)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"查询会话历史异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        session_manager.clear_session(session_id)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.error(f"清除会话异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- 知识图谱管理 APIs ----

@router.post("/knowledge/graph/entity", summary="新增/更新图谱实体")
async def upsert_entity(request: GraphEntityRequest):
    # 手动添加/更新知识图谱中的实体节点（如客户、产品、员工等）
    try:
        neo4j_store.upsert_entity(
            name=request.name,
            entity_type=request.entity_type,
            business_type=request.business_type,
            properties=request.properties,
        )
        return {"status": "success", "name": request.name}
    except Exception as e:
        logger.error(f"图谱实体操作异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/graph/relationship", summary="新建图谱关系")
async def create_relationship(request: GraphRelationshipRequest):
    # 手动在实体之间创建业务关系（如 投诉、购买、负责等）
    try:
        neo4j_store.create_relationship(
            from_name=request.from_entity,
            rel_type=request.rel_type,
            to_name=request.to_entity,
            business_type=request.business_type,
            properties=request.properties,
        )
        return {
            "status": "success",
            "relationship": f"{request.from_entity} -[{request.rel_type}]-> {request.to_entity}",
        }
    except Exception as e:
        logger.error(f"图谱关系操作异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/graph/entities", summary="查询图谱实体列表")
async def list_entities(business_type: Optional[str] = None):
    # 查看当前业务域下所有图谱实体，用于调试和可视化
    try:
        if business_type:
            results = neo4j_store.run_query(
                "MATCH (e:Entity {business_type: $bt}) "
                "RETURN e.name AS name, e.entity_type AS entity_type, "
                "e.business_type AS business_type ORDER BY e.name",
                {"bt": business_type},
            )
        else:
            results = neo4j_store.run_query(
                "MATCH (e:Entity) "
                "RETURN e.name AS name, e.entity_type AS entity_type, "
                "e.business_type AS business_type ORDER BY e.name"
            )
        return {"entities": results, "count": len(results)}
    except Exception as e:
        logger.error(f"查询图谱实体异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/graph/query", summary="执行自定义 Cypher 查询")
async def graph_query(request: GraphQueryRequest):
    # 开放 Cypher 查询接口，方便直接操作图数据库进行调试
    try:
        results = neo4j_store.run_query(request.cypher, request.params or {})
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Cypher 查询异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/graph/stats", summary="图谱统计")
async def graph_stats(business_type: Optional[str] = None):
    # 统计当前业务域下的实体数、知识块数、关系数
    try:
        if business_type:
            entity_count = neo4j_store.run_query(
                "MATCH (e:Entity {business_type: $bt}) RETURN count(e) AS count",
                {"bt": business_type},
            )
            chunk_count = neo4j_store.run_query(
                "MATCH (c:Chunk {business_type: $bt}) RETURN count(c) AS count",
                {"bt": business_type},
            )
            rel_count = neo4j_store.run_query(
                "MATCH ()-[r]->() WHERE $bt IS NULL OR r.business_type = $bt "
                "RETURN count(r) AS count",
                {"bt": business_type},
            )
        else:
            entity_count = neo4j_store.run_query(
                "MATCH (e:Entity) RETURN count(e) AS count"
            )
            chunk_count = neo4j_store.run_query(
                "MATCH (c:Chunk) RETURN count(c) AS count"
            )
            rel_count = neo4j_store.run_query(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            )
        return {
            "business_type": business_type or "all",
            "entities": entity_count[0]["count"] if entity_count else 0,
            "chunks": chunk_count[0]["count"] if chunk_count else 0,
            "relationships": rel_count[0]["count"] if rel_count else 0,
        }
    except Exception as e:
        logger.error(f"图谱统计异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/graph/link", summary="关联实体到知识块")
async def link_entity_to_chunk(
    entity_name: str = Form(...),
    chunk_id: int = Form(...),
    business_type: str = Form(default="default"),
):
    # 建立实体到知识块之间的提及关系，使图检索能找到关联的知识块
    try:
        neo4j_store.link_entity_to_chunk(entity_name, chunk_id, business_type)
        return {
            "status": "success",
            "entity_name": entity_name,
            "chunk_id": chunk_id,
        }
    except Exception as e:
        logger.error(f"关联实体到知识块异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))
