from core.document_loader.base import DocumentLoader, TextLoader, get_loader
from core.document_loader.cleaner import clean_text, filter_short_fragments, deduplicate_chunks
from core.document_loader.pdf_loader import PDFLoader
from core.document_loader.docx_loader import DocxLoader

__all__ = [
    "DocumentLoader",
    "TextLoader",
    "PDFLoader",
    "DocxLoader",
    "get_loader",
    "clean_text",
    "filter_short_fragments",
    "deduplicate_chunks",
]
