"""
Cosine similarity search over the local embedding index.
No Streamlit, no I/O side effects beyond loading the pickle — this keeps
it independently testable (see tests/test_search.py).
"""
import pandas as pd
import numpy as np

from config import LOCAL_INDEX_PATH, TOP_K
from src.embedder import embed_text


def load_index() -> pd.DataFrame:
    return pd.read_pickle(LOCAL_INDEX_PATH)


def search(query: str, df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Embed the query, compute cosine similarity against every row's
    embedding, return the top_k rows sorted by similarity (highest first).
    """
    query_vec = embed_text(query)  # already normalized (see embedder.py)

    # Stack all row embeddings into one (n_rows, 384) matrix
    matrix = np.vstack(df["embedding"].values)

    # Since both query_vec and every row are unit-normalized, cosine
    # similarity IS the dot product — no need for sklearn's
    # cosine_similarity() here, this is faster and makes the math explicit.
    scores = matrix @ query_vec

    result = df.copy()
    result["similarity"] = scores
    result = result.sort_values("similarity", ascending=False).head(top_k)

    return result[["title", "summary", "similarity", "start", "youtube_url", "speaker"]]