"""
Thin wrapper around the local embedding model.
This is the ONLY file in the project that knows which model we're using —
build_index.py and search.py both go through this, never import
sentence_transformers directly. That means swapping models later is a
one-file change.
"""
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Loads the model once and caches it (lru_cache with maxsize=1 means
    the model is loaded on first call, then reused). Loading a model is
    slow (~seconds) and memory-heavy — you never want to reload it per
    request in a real app.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """Embed a single string (e.g. a user's search query)."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of strings (e.g. all transcript segments during index
    build). Batching is much faster than calling embed_text() in a loop —
    the model processes multiple inputs in parallel.
    """
    model = get_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )