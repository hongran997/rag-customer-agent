from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.document_loader.cleaner import deduplicate_chunks
from utils.logger import logger
from utils.constants import CHUNK_SIZE, CHUNK_OVERLAP


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        if separators is None:
            separators = [
                "\n\n## ",
                "\n\n### ",
                "\n\n#### ",
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "，",
                " ",
                "",
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            keep_separator=True,
        )
        logger.info(
            f"语义分块器初始化: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        chunks = [c.strip() for c in chunks if c.strip()]
        chunks = deduplicate_chunks(chunks)

        logger.info(f"文本分块完成: 输入 {len(text)} 字符 → {len(chunks)} 块")
        return chunks

    def split_with_metadata(
        self, text: str, source_doc: str = "", business_type: str = ""
    ) -> List[dict]:
        chunks = self.split_text(text)
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "text_chunk": chunk,
                "chunk_index": i,
                "source_doc": source_doc,
                "business_type": business_type,
            })
        return result
