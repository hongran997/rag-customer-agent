from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="用户问题")
    session_id: str = Field(..., description="会话ID")
    business_type: str = Field(default="default", description="业务类型标签")
    stream: bool = Field(default=False, description="是否流式输出")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="回答内容")
    answer_with_source: str = Field(default="", description="带来源的回答")
    session_id: str = Field(..., description="会话ID")
    intent_type: str = Field(default="business_consult", description="意图类型")
    reference_source: List[str] = Field(default=[], description="参考来源")
    is_hallucination: bool = Field(default=False, description="是否幻觉")


class DocumentUploadRequest(BaseModel):
    business_type: str = Field(default="default", description="业务类型")


class DocumentUploadResponse(BaseModel):
    file: str = Field(..., description="文件名")
    chunks: int = Field(default=0, description="分块数")
    status: str = Field(..., description="状态")


class BatchUploadResponse(BaseModel):
    total_files: int = Field(..., description="总文件数")
    total_chunks: int = Field(..., description="总分块数")
    files: List[Dict[str, Any]] = Field(..., description="文件详情")


class SessionHistoryRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    max_turns: int = Field(default=10, description="最大返回轮数")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(default=None, description="错误详情")
