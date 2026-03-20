import hashlib
import math
from langchain_core.embeddings import Embeddings


class HashingEmbeddings(Embeddings):
    """
    Lightweight deterministic embeddings.
    Good for deployment when you want to avoid heavy ML dependencies.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vectorize(self, text: str):
        vec = [0.0] * self.dim
        tokens = text.lower().split()

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = -1.0 if ((h >> 8) & 1) else 1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text):
        return self._vectorize(text)