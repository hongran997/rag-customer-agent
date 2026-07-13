import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
from core.document_loader.pdf_loader import PDFLoader
from core.document_loader.table_processor import (
    extract_tables_from_page,
    _table_overlap_with_page_text,
    _ensure_markdown_table_header,
    TABLE_DEDUP_THRESHOLD,
)
from core.document_loader.base import get_loader


def _create_table_pdf(data, page_text_before="", x0=72, y0=100, cell_w=150, cell_h=30):
    doc = fitz.open()
    page = doc.new_page()

    if page_text_before:
        page.insert_text(fitz.Point(72, 50), page_text_before, fontsize=11)

    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    x1 = x0 + cell_w * cols
    y1 = y0 + cell_h * rows

    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=0, width=1)
    for r in range(rows + 1):
        y = y0 + r * cell_h
        page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y), color=0, width=1)
    for c in range(cols + 1):
        x = x0 + c * cell_w
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y1), color=0, width=1)

    for r in range(rows):
        for c in range(cols):
            page.insert_text(fitz.Point(x0 + c * cell_w + 5, y0 + r * cell_h + 8),
                           str(data[r][c]), fontsize=10)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    return tmp.name


def test_table_overlap():
    ratio_high = _table_overlap_with_page_text(
        "|Name|Price|\n|---|---|\n|A|100|",
        "Product Name: A, Price: 100",
    )
    assert ratio_high > 0.6, f"expected >0.6, got {ratio_high}"
    ratio_low = _table_overlap_with_page_text(
        "|X12|Y78|\n|---|---|\n|Z34|W56|",
        "this is a page of random english text with common letters",
    )
    assert ratio_low < 0.5, f"expected <0.5, got {ratio_low}"
    assert _table_overlap_with_page_text("", "anything") == 0.0
    print(f"[OK] _table_overlap_with_page_text (threshold={TABLE_DEDUP_THRESHOLD})")


def test_ensure_markdown_table_header():
    input_already_ok = "|Name|Price|\n|---|---|\n|A|100|"
    result = _ensure_markdown_table_header(input_already_ok)
    assert result == input_already_ok

    input_no_sep = "|Name|Price|\n|A|100|"
    result = _ensure_markdown_table_header(input_no_sep)
    assert "|---|---|" in result
    assert result.count("|") == 9

    input_single = "|Name|"
    result = _ensure_markdown_table_header(input_single)
    assert result == "|Name|"

    print("[OK] _ensure_markdown_table_header")


def test_extract_tables_basic():
    data = [["Name", "Price"], ["ProductA", "100"], ["ProductB", "200"]]
    pdf_path = _create_table_pdf(data)

    doc = fitz.open(pdf_path)
    page = doc[0]
    result = extract_tables_from_page(page, "")
    doc.close()

    assert "[表格]" in result
    assert "[/表格]" in result
    assert "Name" in result
    assert "Price" in result
    assert "ProductA" in result
    assert "ProductB" in result
    print(f"[OK] extract_tables_basic:\n{result}")
    os.unlink(pdf_path)


def test_extract_tables_dedup():
    data = [["Name", "Price"], ["A", "100"]]
    pdf_path = _create_table_pdf(data)

    doc = fitz.open(pdf_path)
    page = doc[0]
    existing_text = "Name Price A 100"
    result = extract_tables_from_page(page, existing_text)
    doc.close()

    assert result == "", f"expected empty (dedup), got: {result}"
    print("[OK] extract_tables_dedup: 表格与已有文本重复，正确跳过")
    os.unlink(pdf_path)


def test_pdf_loader_with_table():
    data = [["Model", "Power", "Price"], ["X100", "500W", "299"], ["X200", "1000W", "499"]]
    pdf_path = _create_table_pdf(data)

    loader = PDFLoader(pdf_path)
    text = loader.load()
    meta = loader.get_metadata()

    assert meta["page_count"] == 1
    assert "[表格]" in text
    assert "[/表格]" in text
    assert "Model" in text
    assert "500W" in text
    assert "Power" in text
    print(f"[OK] PDFLoader with table: {len(text)} chars, has marker: {'[表格]' in text}")
    os.unlink(pdf_path)


def test_pdf_loader_with_text_and_table():
    data = [["Param", "Value"], ["Voltage", "220V"], ["Freq", "50Hz"]]
    pdf_path = _create_table_pdf(data)
    loader = PDFLoader(pdf_path)
    text = loader.load()
    assert "[表格]" in text
    assert "Param" in text
    assert "Voltage" in text
    print(f"[OK] PDFLoader with text + table: {len(text)} chars")
    os.unlink(pdf_path)


def test_get_loader_with_table():
    data = [["Item", "Value"], ["Temp", "25C"]]
    pdf_path = _create_table_pdf(data)

    loader = get_loader(pdf_path)
    text = loader.load()
    meta = loader.get_metadata()

    assert meta["file_type"] == ".pdf"
    assert "[表格]" in text
    print(f"[OK] get_loader with table: {meta}")
    os.unlink(pdf_path)


def test_table_with_same_line_text_dedup():
    data = [["Name", "Price"], ["ItemA", "99"]]
    pdf_path = _create_table_pdf(data)

    doc = fitz.open(pdf_path)
    page = doc[0]

    page_text = page.get_text("text")
    result = extract_tables_from_page(page, page_text)
    doc.close()

    assert result == "", f"table content in get_text, should be deduped: {result}"
    print("[OK] table_with_same_line_text_dedup: get_text 已包含表格文字，正确跳过")
    os.unlink(pdf_path)


if __name__ == "__main__":
    test_table_overlap()
    test_ensure_markdown_table_header()
    test_extract_tables_basic()
    test_extract_tables_dedup()
    test_table_with_same_line_text_dedup()
    test_pdf_loader_with_table()
    test_pdf_loader_with_text_and_table()
    test_get_loader_with_table()
    print("\n所有表格测试通过!")
