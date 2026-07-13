import re
import fitz
from core.document_loader.base import DocumentLoader
from core.document_loader.cleaner import clean_text
from core.document_loader.ocr_processor import ocr_image_bytes, should_skip_image
from core.document_loader.table_processor import extract_tables_from_page
from utils.logger import logger

OCR_DEDUP_THRESHOLD = 0.6


def _ocr_overlap_with_page_text(ocr_text: str, page_text: str) -> float:
    ocr_chars = set(re.sub(r'\s+', '', ocr_text))
    page_chars = set(re.sub(r'\s+', '', page_text))
    if not ocr_chars:
        return 0.0
    intersection = ocr_chars & page_chars
    return len(intersection) / len(ocr_chars)


class PDFLoader(DocumentLoader):
    def load(self) -> str:
        text_parts = []
        total_ocr_chars = 0
        total_table_chars = 0
        doc = fitz.open(self.file_path)
        logger.info(f"解析PDF文档: {self.file_path.name}, 共 {len(doc)} 页")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            page_text = clean_text(page_text)

            table_text = extract_tables_from_page(page, page_text)
            existing_for_ocr = page_text
            if table_text:
                existing_for_ocr += "\n\n" + table_text

            ocr_texts = self._extract_images_text(doc, page, page_num, existing_for_ocr)

            combined = page_text
            if table_text:
                combined += "\n\n" + table_text
                total_table_chars += len(table_text)
            if ocr_texts:
                combined += "\n\n" + ocr_texts
                total_ocr_chars += len(ocr_texts)
            if combined.strip():
                text_parts.append(combined)
        doc.close()
        full_text = "\n\n".join(text_parts)
        logger.info(
            f"PDF解析完成: {self.file_path.name}, "
            f"提取字符数: {len(full_text)} "
            f"(其中表格: {total_table_chars} 字符, "
            f"OCR识别: {total_ocr_chars} 字符)"
        )
        return full_text

    def _extract_images_text(self, doc: fitz.Document, page: fitz.Page, page_num: int, existing_text: str) -> str:
        image_list = page.get_images(full=True)
        if not image_list:
            return ""
        seen_xrefs = set()
        ocr_parts = []
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            width = img_info[2]
            height = img_info[3]
            if should_skip_image(width, height):
                continue
            try:
                img_data = doc.extract_image(xref)
                img_bytes = img_data.get("image")
                if not img_bytes:
                    continue
                text = ocr_image_bytes(img_bytes)
                if not text.strip():
                    continue
                if _ocr_overlap_with_page_text(text, existing_text) > OCR_DEDUP_THRESHOLD:
                    logger.info(
                        f"第 {page_num + 1} 页图片(xref={xref})内容与已有文本高度重复，跳过 OCR"
                    )
                    continue
                ocr_parts.append(text)
            except Exception as e:
                logger.warning(
                    f"第 {page_num + 1} 页提取图片(xref={xref})失败: {e}"
                )
                continue
        if not ocr_parts:
            return ""
        return "[图片文字]\n" + "\n\n".join(ocr_parts) + "\n[/图片文字]"

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
