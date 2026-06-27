from typing import Generator, AsyncGenerator
from fastapi.responses import StreamingResponse
from utils.logger import logger


def stream_response(
    generator: Generator[str, None, None],
) -> StreamingResponse:
    async def event_stream():
        try:
            for chunk in generator:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式输出异常: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
