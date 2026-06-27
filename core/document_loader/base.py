from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class DocumentLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    @abstractmethod
    def load(self) -> str:
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        ...


class TextLoader(DocumentLoader):
    def load(self) -> str:
        return self._read_file()

    def get_metadata(self) -> dict:
        return {
            "source_doc": self.file_path.name,
            "file_type": self.file_path.suffix.lower(),
            "file_size": self.file_path.stat().st_size,
        }

    def _read_file(self) -> str:
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def get_loader(file_path: str) -> DocumentLoader:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from core.document_loader.pdf_loader import PDFLoader
        return PDFLoader(file_path)
    elif suffix == ".docx":
        from core.document_loader.docx_loader import DocxLoader
        return DocxLoader(file_path)
    elif suffix in (".md", ".markdown"):
        from core.document_loader.md_loader import MarkdownLoader
        return MarkdownLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")
