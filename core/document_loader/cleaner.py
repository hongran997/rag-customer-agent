import re
from utils.logger import logger
from utils.constants import MIN_TEXT_LENGTH


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = _remove_header_footer(line)
        line = _remove_page_numbers(line)
        line = _remove_special_chars(line)
        line = _remove_watermark(line)
        line = line.strip()

        if not line:
            continue

        if len(line) < MIN_TEXT_LENGTH and not re.search(r'[。！？：；\n]', line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _remove_duplicate_blank_lines(text)
    text = _remove_garbled_text(text)

    return text.strip()


def _remove_header_footer(line: str) -> str:
    patterns = [
        r'^\d+/\d+$',
        r'^第\s*\d+\s*页$',
        r'^-\s*\d+\s*-$',
        r'^Page\s+\d+',
        r'^Copyright\s+\d{4}',
        r'^All\s+Rights\s+Reserved',
        r'^[A-Z\s]{10,}$',
        r'^公司[^\w]{0,10}(简介|概况|介绍)',
    ]
    for pat in patterns:
        if re.match(pat, line.strip(), re.IGNORECASE):
            return ""
    return line


def _remove_page_numbers(line: str) -> str:
    line = re.sub(r'\f', '', line)
    return line


def _remove_special_chars(line: str) -> str:
    line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', line)
    line = re.sub(r'[●■◆▲▼※→⇒♻✅☑✘✗✓✕]', '', line)
    return line


def _remove_watermark(line: str) -> str:
    watermark_patterns = [
        r'仅供内部使用',
        r'机密文件',
        r'Confidential',
        r'Draft',
        r'草稿',
        r'水印',
    ]
    for pat in watermark_patterns:
        if re.search(pat, line.strip(), re.IGNORECASE):
            return ""
    return line


def _remove_duplicate_blank_lines(text: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', text)


def _remove_garbled_text(text: str) -> str:
    garbled_pattern = re.compile(
        r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
        r'a-zA-Z0-9\s.,!?;:()（）\[\]【】{}《》<>""''＂＇'
        r'+\-*/=#@&%$￥¥€£©®™℃°·×÷→←↑↓\'\"]'
    )
    text = garbled_pattern.sub('', text)
    return text


def filter_short_fragments(chunks: list[str]) -> list[str]:
    return [c for c in chunks if len(c.strip()) >= MIN_TEXT_LENGTH]


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    seen = set()
    result = []
    for c in chunks:
        normalized = re.sub(r'\s+', '', c)
        if normalized not in seen:
            seen.add(normalized)
            result.append(c)
    return result


logger.info("文本清洗模块加载完成")
