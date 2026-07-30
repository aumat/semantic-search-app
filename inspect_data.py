import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "embedding_index_3m.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type of top-level object:", type(data))
print("Number of records:", len(data))
print("\nFirst record's keys:")
print(list(data[0].keys()))
print("\nFirst record (embedding truncated):")
first = dict(data[0])
for k, v in first.items():
    if isinstance(v, list) and len(v) > 10:
        first[k] = f"<list of {len(v)} floats>"
print(first)