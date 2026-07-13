import re
import fitz
from core.document_loader.base import DocumentLoader
from core.document_loader.cleaner import clean_text
from core.document_loader.ocr_processor import ocr_image_bytes, should_skip_image
from core.document_loader.table_processor import extract_tables_structured, merge_cross_page_tables
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
        table_merge_count = 0
        # 跨页表格追踪：记录上一页最后一个表格的结构
        pending_table = None  # {cols, header, sep, rows, text_parts_idx}

        doc = fitz.open(self.file_path)
        logger.info(f"解析PDF文档: {self.file_path.name}, 共 {len(doc)} 页")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            page_text = clean_text(page_text)

            table_text, table_metas = extract_tables_structured(page, page_text)
            current_table_merged = False

            if pending_table is not None and table_metas:
                first = table_metas[0]
                if first["cols"] == pending_table["cols"]:
                    # 跨页续表：该表格首行在 find_tables() 中被当成了表头，
                    # 实际是数据行的延续，需要一并合并
                    if first["header"]:
                        pending_table["rows"].append(first["header"])
                    pending_table["rows"].extend(first["rows"])
                    merged_lines = [pending_table["header"]]
                    if pending_table["sep"]:
                        merged_lines.append(pending_table["sep"])
                    merged_lines.extend(pending_table["rows"])
                    merged_md = "\n".join(merged_lines)
                    merged_block = f"[表格]\n{merged_md}\n[/表格]"

                    # 更新 text_parts 中上一页的表格内容
                    old_entry = text_parts[pending_table["text_parts_idx"]]
                    old_entry_updated = re.sub(
                        r'\[表格\].*?\[/表格\]',
                        lambda m: merged_block,
                        old_entry,
                        count=1,
                        flags=re.DOTALL,
                    )
                    text_parts[pending_table["text_parts_idx"]] = old_entry_updated
                    table_merge_count += 1

                    # 当前页去掉已合并的第一个表格
                    remaining = table_metas[1:]
                    if remaining:
                        remaining_raw = [m["raw"] for m in remaining]
                        table_text = "[表格]\n" + "\n\n".join(remaining_raw) + "\n[/表格]"
                    else:
                        table_text = ""
                    current_table_merged = True

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

            # 更新跨页追踪状态
            if table_metas:
                if current_table_merged:
                    # 第一个表格已合并到上页，追踪本页最后一个（若有剩余表格）
                    last_meta = table_metas[-1]
                    pending_table = {
                        "cols": last_meta["cols"],
                        "header": last_meta["header"],
                        "sep": last_meta["sep"],
                        "rows": list(last_meta["rows"]),
                        "text_parts_idx": len(text_parts) - 1,
                    }
                else:
                    last_meta = table_metas[-1]
                    pending_table = {
                        "cols": last_meta["cols"],
                        "header": last_meta["header"],
                        "sep": last_meta["sep"],
                        "rows": list(last_meta["rows"]),
                        "text_parts_idx": len(text_parts) - 1,
                    }
            else:
                pending_table = None

        doc.close()
        full_text = "\n\n".join(text_parts)
        # 再对白空格相连的表格做一次后处理合并
        merged_text = merge_cross_page_tables(full_text)
        merged_count = merged_text.count("[表格]")
        before_count = full_text.count("[表格]")
        total_merged = table_merge_count + (before_count - merged_count)
        logger.info(
            f"PDF解析完成: {self.file_path.name}, "
            f"提取字符数: {len(merged_text)} "
            f"(其中表格: {total_table_chars} 字符, "
            f"OCR识别: {total_ocr_chars} 字符, "
            f"跨页合并: {total_merged} 个)"
        )
        return merged_text

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
