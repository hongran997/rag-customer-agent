from typing import List, Optional
from pathlib import Path
import numpy as np
from core.document_loader import get_loader, deduplicate_chunks, filter_short_fragments
from core.chunker import SemanticChunker
from core.embedding import embedding_model
from core.vector_store import milvus_store
from utils.logger import logger


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def process_single_document(
    file_path: str,
    business_type: str = "default",
) -> List[dict]:
    logger.info(f"开始处理文档: {file_path}")
    loader = get_loader(file_path)
    raw_text = loader.load()

    if not raw_text or not raw_text.strip():
        logger.warning(f"文档内容为空: {file_path}")
        return []

    chunker = SemanticChunker()
    chunks_with_meta = chunker.split_with_metadata(
        text=raw_text,
        source_doc=loader.get_metadata()["source_doc"],
        business_type=business_type,
    )

    if not chunks_with_meta:
        logger.warning(f"文档分块后为空: {file_path}")
        return []

    texts = [c["text_chunk"] for c in chunks_with_meta]
    vectors = embedding_model.encode(texts)

    milvus_store.insert(
        texts=texts,
        vectors=vectors,
        source_docs=[c["source_doc"] for c in chunks_with_meta],
        business_types=[c["business_type"] for c in chunks_with_meta],
        chunk_indices=[c["chunk_index"] for c in chunks_with_meta],
    )

    logger.info(
        f"文档处理完成: {file_path} → {len(chunks_with_meta)} 块向量入库"
    )
    return chunks_with_meta


def process_folder(
    folder_path: str,
    business_type: str = "default",
    extensions: Optional[set] = None,
) -> dict:
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"文件夹不存在: {folder_path}")

    all_files = [
        f for f in folder.rglob("*")
        if f.suffix.lower() in extensions and not f.name.startswith(".")
    ]

    if not all_files:
        logger.warning(f"文件夹中未找到支持的文档: {folder_path}")
        return {"total_files": 0, "total_chunks": 0, "files": []}

    stats = {"total_files": len(all_files), "total_chunks": 0, "files": []}
    for file_path in all_files:
        try:
            chunks = process_single_document(
                str(file_path), business_type=business_type
            )
            stats["total_chunks"] += len(chunks)
            stats["files"].append({
                "file": file_path.name,
                "chunks": len(chunks),
                "status": "success",
            })
        except Exception as e:
            logger.error(f"文档处理失败: {file_path} - {str(e)}")
            stats["files"].append({
                "file": file_path.name,
                "chunks": 0,
                "status": f"failed: {str(e)}",
            })

    logger.info(
        f"批量入库完成: {stats['total_files']} 文件, "
        f"{stats['total_chunks']} 分块"
    )
    return stats
