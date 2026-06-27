import fitz
from core.document_loader.base import DocumentLoader
from core.document_loader.cleaner import clean_text
from utils.logger import logger


class PDFLoader(DocumentLoader):
    def load(self) -> str:
        text_parts = []
        doc = fitz.open(self.file_path)
        logger.info(f"解析PDF文档: {self.file_path.name}, 共 {len(doc)} 页")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            page_text = clean_text(page_text)
            if page_text.strip():
                text_parts.append(page_text)
        doc.close()
        full_text = "\n\n".join(text_parts)
        logger.info(
            f"PDF解析完成: {self.file_path.name}, "
            f"提取字符数: {len(full_text)}"
        )
        return full_text

    def get_metadata(self) -> dict:
        doc = fitz.open(self.file_path)
        meta = {
            "source_doc": self.file_path.name,
            "file_type": ".pdf",
            "file_size": self.file_path.stat().st_size,
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
        }
        doc.close()
        return meta
