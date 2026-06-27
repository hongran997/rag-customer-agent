import markdown
import re
from core.document_loader.base import TextLoader
from core.document_loader.cleaner import clean_text
from utils.logger import logger


class MarkdownLoader(TextLoader):
    def load(self) -> str:
        raw_text = super().load()
        html = markdown.markdown(raw_text, extensions=["extra"])
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
        text = clean_text(text)
        logger.info(
            f"MD解析完成: {self.file_path.name}, "
            f"提取字符数: {len(text)}"
        )
        return text

    def get_metadata(self) -> dict:
        meta = super().get_metadata()
        meta["file_type"] = ".md"
        return meta
