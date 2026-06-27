from fastapi import Request
from fastapi.responses import JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
from utils.logger import logger
import asyncio


TIMEOUT_SECONDS = 15


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
async def with_retry(coro, name: str = "operation"):
    try:
        return await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning(f"{name} 超时, 即将重试")
        raise TimeoutError(f"{name} timeout")
    except ConnectionError:
        logger.warning(f"{name} 连接错误, 即将重试")
        raise


async def global_timeout_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await asyncio.wait_for(
            call_next(request), timeout=TIMEOUT_SECONDS + 5
        )
        elapsed = time.time() - start_time
        logger.debug(f"{request.method} {request.url.path} - {elapsed:.2f}s")
        return response
    except asyncio.TimeoutError:
        logger.error(f"请求超时: {request.method} {request.url.path}")
        return JSONResponse(
            status_code=504,
            content={"error": "请求超时", "detail": f"超过 {TIMEOUT_SECONDS}s 未响应"},
        )
    except Exception as e:
        logger.error(f"请求异常: {request.method} {request.url.path} - {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "服务器内部错误", "detail": str(e)},
        )
