from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "embedding_index_3m.json"
LOCAL_INDEX_PATH = PROJECT_ROOT / "data" / "local_index.pkl"

# Embedding model — small, fast, CPU-friendly, good default for semantic search
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Search behavior
TOP_K = 5