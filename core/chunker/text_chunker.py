import re
from typing import List, Optional
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.document_loader.cleaner import deduplicate_chunks
from utils.logger import logger
from utils.constants import CHUNK_SIZE, CHUNK_OVERLAP

# 需要保护的结构化标记块（不分块时切开）
_PROTECTED_BLOCKS = [
    (r'\[表格\].*?\[/表格\]', '[表格]'),
    (r'\[图片文字\].*?\[/图片文字\]', '[图片文字]'),
]


def _protect_blocks(text: str):
    placeholders = {}
    counter = [0]

    def _replace(m: re.Match, tag: str):
        idx = counter[0]
        counter[0] += 1
        key = f"\x00BLOCK_{idx}_{tag}\x00"
        placeholders[key] = m.group(0)
        return key

    protected = text
    for pattern, tag in _PROTECTED_BLOCKS:
        protected = re.sub(
            pattern,
            lambda m, t=tag: _replace(m, t),
            protected,
            flags=re.DOTALL,
        )

    return protected, placeholders


def _restore_blocks(chunks: List[str], placeholders: dict) -> List[str]:
    result = []
    for chunk in chunks:
        restored = chunk
        for key, value in placeholders.items():
            if key in restored:
                restored = restored.replace(key, value)
        result.append(restored)
    return result


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

        protected_text, placeholders = _protect_blocks(text)
        chunks = self.splitter.split_text(protected_text)
        chunks = [c.strip() for c in chunks if c.strip()]
        chunks = _restore_blocks(chunks, placeholders)
        chunks = deduplicate_chunks(chunks)

        protected_count = len(placeholders)
        logger.info(
            f"文本分块完成: 输入 {len(text)} 字符 → {len(chunks)} 块 "
            f"(保护 {protected_count} 个结构化块)"
        )
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
