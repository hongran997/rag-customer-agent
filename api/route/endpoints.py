from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from api.route.models import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    BatchUploadResponse,
    SessionHistoryRequest,
    ErrorResponse,
)
from core.agent.rag_agent import rag_agent
from core.pipeline import process_single_document, process_folder
from api.session_store import session_manager
from api.stream.stream_helper import stream_response
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
    try:
        from core.vector_store import milvus_store
        milvus_store.delete_by_business_type(business_type)
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
