from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import logger
from utils.constants import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_DIM


class BGEEmbedding:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        logger.info(f"加载嵌入模型: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = EMBEDDING_DIM
        self._cache = {}
        logger.info(f"嵌入模型加载完成, 向量维度: {self.dim}")

    def encode(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        uncached_texts = []
        uncached_indices = []
        embeddings_list = [None] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings_list[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            if normalize:
                embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )

            for idx, text, emb in zip(
                uncached_indices, uncached_texts, embeddings
            ):
                self._cache[text] = emb
                embeddings_list[idx] = emb

        return np.array(embeddings_list)

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query, normalize_embeddings=True
        )
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


embedding_model = BGEEmbedding()
