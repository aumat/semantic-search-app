"""
One-time script: reads embedding_index_3m.json, re-embeds every segment's
summary with our LOCAL model (discarding the OpenAI 'ada_v2' vectors), and
saves the result to data/local_index.pkl.

Run this ONCE after any change to the dataset or the embedding model:
    python -m src.build_index
"""
import json
import pandas as pd

from config import RAW_DATA_PATH, LOCAL_INDEX_PATH, EMBEDDING_MODEL_NAME
from src.embedder import embed_texts


def build_youtube_url(video_id: str, seconds: int) -> str:
    return f"https://youtube.com/watch?v={video_id}&t={seconds}s"


def main():
    print(f"Loading raw data from {RAW_DATA_PATH} ...")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records.")

    df = pd.DataFrame(records)

    # Drop the OpenAI embedding column entirely — different model, different
    # vector space, not usable here. Keeping it around would only invite
    # someone (future you, six months from now) to accidentally mix vector
    # spaces later.
    df = df.drop(columns=["ada_v2"])

    # We embed the summary — it's the only descriptive text field available.
    print(f"Embedding {len(df)} summaries with '{EMBEDDING_MODEL_NAME}' "
          f"(this may take a minute or two on CPU)...")
    embeddings = embed_texts(df["summary"].tolist())

    df["embedding"] = list(embeddings)  # one row -> one embedding vector
    df["youtube_url"] = df.apply(
        lambda r: build_youtube_url(r["videoId"], r["seconds"]), axis=1
    )

    LOCAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(LOCAL_INDEX_PATH)
    print(f"Saved local index ({len(df)} rows) to {LOCAL_INDEX_PATH}")


if __name__ == "__main__":
    main()