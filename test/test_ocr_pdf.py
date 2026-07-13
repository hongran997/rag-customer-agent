import sys, os, io, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
from core.document_loader.pdf_loader import PDFLoader, _ocr_overlap_with_page_text, OCR_DEDUP_THRESHOLD
from core.document_loader.base import get_loader
from core.document_loader.ocr_processor import (
    should_skip_image, OCR_IMAGE_MIN_SIZE, OCR_CONFIDENCE_THRESHOLD
)


def test_should_skip_image():
    assert should_skip_image(10, 10) is True
    assert should_skip_image(63, 100) is True
    assert should_skip_image(100, 63) is True
    assert should_skip_image(64, 64) is False
    assert should_skip_image(200, 200) is False
    print("[OK] should_skip_image")


def test_ocr_overlap():
    ratio_high = _ocr_overlap_with_page_text("系统配置说明", "这是一段系统配置说明文字")
    assert ratio_high > 0.6, f"expected >0.6, got {ratio_high}"
    ratio_low = _ocr_overlap_with_page_text("HelloWorld", "中文文本内容")
    assert ratio_low == 0.0, f"expected 0.0, got {ratio_low}"
    assert _ocr_overlap_with_page_text("", "anything") == 0.0
    print(f"[OK] _ocr_overlap_with_page_text (threshold={OCR_DEDUP_THRESHOLD})")


def test_constants():
    assert OCR_IMAGE_MIN_SIZE == 64
    assert OCR_CONFIDENCE_THRESHOLD == 0.3
    assert OCR_DEDUP_THRESHOLD == 0.6
    print("[OK] constants")


def test_metadata():
    doc = fitz.open()
    page = doc.new_page()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    loader = get_loader(tmp.name)
    meta = loader.get_metadata()
    assert meta["source_doc"] == os.path.basename(tmp.name)
    assert meta["file_type"] == ".pdf"
    assert meta["page_count"] == 1
    print(f"[OK] metadata: {meta}")
    os.unlink(tmp.name)


def test_empty_pdf():
    doc = fitz.open()
    page = doc.new_page()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    loader = PDFLoader(tmp.name)
    text = loader.load()
    assert isinstance(text, str)
    print(f"[OK] empty PDF: {len(text)} chars")
    os.unlink(tmp.name)


def test_image_only_pdf_loads():
    from PIL import Image, ImageDraw
    doc = fitz.open()
    page = doc.new_page()
    w, h = 200, 200
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "图表标题", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rect = fitz.Rect(50, 50, 250, 250)
    page.insert_image(rect, stream=buf.getvalue())
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    loader = PDFLoader(tmp.name)
    text = loader.load()
    assert isinstance(text, str)
    print(f"[OK] image-only PDF: {len(text)} chars (OCR status depends on network)")
    os.unlink(tmp.name)


if __name__ == "__main__":
    test_should_skip_image()
    test_constants()
    test_ocr_overlap()
    test_metadata()
    test_empty_pdf()
    test_image_only_pdf_loads()
    print("\n所有测试通过!")
