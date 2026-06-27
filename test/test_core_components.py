import pytest
from core.document_loader.base import get_loader
from core.document_loader.cleaner import clean_text, filter_short_fragments, deduplicate_chunks
from core.chunker import SemanticChunker
import tempfile
import os


class TestDocumentCleaner:
    def test_clean_removes_header_footer(self):
        text = "第 1 页\n这是正文内容。\nCopyright 2024"
        result = clean_text(text)
        assert "第 1 页" not in result
        assert "这是正文内容" in result

    def test_clean_removes_special_chars(self):
        text = "● 产品介绍\n◆ 功能说明"
        result = clean_text(text)
        assert "●" not in result
        assert "产品介绍" in result

    def test_clean_removes_watermark(self):
        text = "这是正文仅供内部使用继续"
        result = clean_text(text)
        assert "仅供内部使用" not in result

    def test_filter_short_fragments(self):
        chunks = ["短", "这是一个大于十个字符的片段内容"]
        result = filter_short_fragments(chunks)
        assert len(result) == 1

    def test_deduplicate(self):
        chunks = ["这是内容", "这是内容", "这是不同内容"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2


class TestDocumentLoader:
    def test_get_loader_unsupported_format(self):
        with pytest.raises(ValueError, match="不支持的文件格式"):
            get_loader("test.xlsx")

    def test_get_loader_txt(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("测试文本内容")
            tmp_path = f.name

        loader = get_loader(tmp_path)
        content = loader.load()
        assert "测试文本内容" in content
        os.unlink(tmp_path)

    def test_get_loader_file_not_found(self):
        from core.document_loader.base import DocumentLoader
        with pytest.raises(FileNotFoundError):
            get_loader("/nonexistent/file.txt")


class TestChunker:
    def test_split_text(self):
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        text = "这是第一段内容。" * 20
        chunks = chunker.split_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text(self):
        chunker = SemanticChunker()
        chunks = chunker.split_text("")
        assert chunks == []

    def test_short_text(self):
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=120)
        chunks = chunker.split_text("短文本")
        assert len(chunks) == 1

    def test_split_with_metadata(self):
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        text = "产品使用说明。" * 10
        result = chunker.split_with_metadata(text, "doc.pdf", "product")
        assert len(result) > 0
        assert result[0]["source_doc"] == "doc.pdf"
        assert result[0]["business_type"] == "product"
