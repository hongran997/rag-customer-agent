import re
from utils.logger import logger

TABLE_DEDUP_THRESHOLD = 0.6


def _table_overlap_with_page_text(table_text: str, page_text: str) -> float:
    table_chars = set(re.sub(r'\s+', '', table_text))
    page_chars = set(re.sub(r'\s+', '', page_text))
    if not table_chars:
        return 0.0
    intersection = table_chars & page_chars
    return len(intersection) / len(table_chars)


def extract_tables_from_page(page, existing_text: str) -> str:
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.warning(f"页面表格检测失败: {e}")
        return ""

    if not tables or not tables.tables:
        return ""

    table_parts = []
    for i, table in enumerate(tables.tables):
        try:
            md_table = table.to_markdown()
            if not md_table.strip():
                continue
            if _table_overlap_with_page_text(md_table, existing_text) > TABLE_DEDUP_THRESHOLD:
                logger.info(f"表格内容与已有文本高度重复，跳过")
                continue
            # 补齐表头分隔行（to_markdown 在有线表格下可能不输出分隔行）
            md_table = _ensure_markdown_table_header(md_table)
            table_parts.append(md_table)
        except Exception as e:
            logger.warning(f"第 {i + 1} 个表格格式化失败: {e}")
            continue

    if not table_parts:
        return ""

    result = "[表格]\n" + "\n\n".join(table_parts) + "\n[/表格]"
    logger.info(f"提取到 {len(table_parts)} 个表格")
    return result


def _ensure_markdown_table_header(md: str) -> str:
    lines = md.strip().split("\n")
    if len(lines) < 2:
        return md
    has_separator = any("---" in line for line in lines)
    if not has_separator and "|" in lines[0]:
        pipe_count = lines[0].count("|")
        separator = "|" + "---|" * (pipe_count - 1)
        lines.insert(1, separator)
        return "\n".join(lines)
    return md
