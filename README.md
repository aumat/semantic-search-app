# Semantic Video Search — Microsoft AI Show

I built this to solve an actual annoyance: our AI Show video library has
hundreds of episodes, and if you vaguely remember "someone explained Jupyter
Notebooks somewhere in there," keyword search is useless — the words you'd
search for often aren't the words used on screen. This searches by *meaning*
instead, so "Can you use RStudio with Azure ML?" correctly surfaces the R
support episodes even though the phrasing doesn't match exactly.

## How it works

1. Every video was already pre-chunked into 3-minute segments with a short
   summary (that part came with the dataset — `embedding_index_3m.json`).
2. Each summary gets converted into a 384-number vector ("embedding") using
   a local model (`all-MiniLM-L6-v2`, via `sentence-transformers`). Similar
   meanings end up as similar vectors.
3. When you type a question, it gets embedded the same way, and we find the
   5 segments whose vectors are closest to it (cosine similarity).
4. Results link straight to the moment in the video, via YouTube's `&t=`
   timestamp parameter.

## Why local embeddings instead of OpenAI

The dataset actually ships with OpenAI embeddings already computed (the
`ada_v2` column). I didn't use them — deliberately went local instead, no
API key, no per-query cost, nothing leaving the machine. The tradeoff:
scores run a bit lower than what you'd see with OpenAI's larger models
(0.6ish for a strong match here, vs. 0.8+ typical with `ada_v2`), so don't
read the raw number as a percentage — it's a *ranking* signal, not a
confidence score.

**One thing to remember if this ever grows:** the local vectors and the
OpenAI vectors are not interchangeable. If this dataset ever gets swapped
for a shared vector database built with OpenAI embeddings, the index here
needs to be rebuilt from scratch with the same model used everywhere — you
can't mix the two in one search.

## Running it

```bash
pip install -r requirements.txt
# place embedding_index_3m.json in data/
python -m src.build_index     # one-time: builds the local embedding index (~1 min)
streamlit run app.py
```
