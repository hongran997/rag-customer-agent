import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from utils.logger import logger
from utils.constants import APP_HOST, APP_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("rag_customer_agent 服务启动中...")
    yield
    logger.info("rag_customer_agent 服务已关闭")


app = FastAPI(
    title="客服知识库智能问答 Agent",
    description="基于 RAG 架构的智能客服问答系统，支持文档入库、混合检索、幻觉约束 Agent 推理、流式问答",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "rag_customer_agent"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
        log_level="info",
    )
