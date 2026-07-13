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


def extract_tables_structured(page, existing_text: str):
    """与 extract_tables_from_page 类似，但额外返回结构化元数据用于跨页合并。

    Returns:
        (table_markdown: str, table_metas: list[dict])
        其中 table_metas 每个元素为 { "cols", "header", "sep", "rows" }
    """
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.warning(f"页面表格检测失败: {e}")
        return "", []

    if not tables or not tables.tables:
        return "", []

    md_parts = []
    metas = []
    for table in tables.tables:
        try:
            md = table.to_markdown()
            if not md.strip():
                continue
            if _table_overlap_with_page_text(md, existing_text) > TABLE_DEDUP_THRESHOLD:
                logger.info(f"表格内容与已有文本高度重复，跳过")
                continue
            md = _ensure_markdown_table_header(md)
            header, sep, rows = _separate_header_and_rows(md)
            cols = _count_md_columns(header)
            md_parts.append(md)
            metas.append({"cols": cols, "header": header, "sep": sep, "rows": list(rows), "raw": md})
        except Exception as e:
            logger.warning(f"表格解析失败: {e}")
            continue

    if not md_parts:
        return "", []

    result = "[表格]\n" + "\n\n".join(md_parts) + "\n[/表格]"
    logger.info(f"提取到 {len(md_parts)} 个表格")
    return result, metas


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


def _parse_table_blocks(text: str):
    pattern = re.compile(
        r'\[表格\]\s*\n(.*?)\n\s*\[/表格\]',
        re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(text):
        blocks.append({
            "start": m.start(),
            "end": m.end(),
            "content": m.group(1).strip(),
        })
    return blocks


def _count_md_columns(line: str) -> int:
    if "|" not in line:
        return 0
    return line.count("|") - 1


def _separate_header_and_rows(content: str):
    lines = content.strip().split("\n")
    header = ""
    sep = ""
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if not header:
            header = stripped
        elif not sep and "---" in stripped:
            sep = stripped
        else:
            rows.append(stripped)
    return header, sep, rows


def merge_cross_page_tables(full_text: str) -> str:
    if "[表格]" not in full_text:
        return full_text

    blocks = _parse_table_blocks(full_text)
    if len(blocks) < 2:
        return full_text

    result_parts = []
    cursor = 0
    i = 0
    while i < len(blocks):
        block = blocks[i]
        gap = full_text[cursor : block["start"]]

        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            gap_to_next = full_text[block["end"] : next_block["start"]]
            only_whitespace = not gap_to_next.strip()

            if only_whitespace:
                header1, sep1, rows1 = _separate_header_and_rows(block["content"])
                header2, sep2, rows2 = _separate_header_and_rows(next_block["content"])
                cols1 = _count_md_columns(header1)
                cols2 = _count_md_columns(header2)

                if cols1 > 0 and cols1 == cols2:
                    merged_lines = [header1]
                    if sep1:
                        merged_lines.append(sep1)
                    merged_lines.extend(rows1)
                    if header2:
                        merged_lines.append(header2)
                    merged_lines.extend(rows2)
                    merged_content = "\n".join(merged_lines)

                    result_parts.append(gap)
                    result_parts.append(f"[表格]\n{merged_content}\n[/表格]")
                    cursor = next_block["end"]
                    i += 2
                    continue

        result_parts.append(gap)
        result_parts.append(full_text[block["start"] : block["end"]])
        cursor = block["end"]
        i += 1

    if cursor < len(full_text):
        result_parts.append(full_text[cursor:])

    return "".join(result_parts)
