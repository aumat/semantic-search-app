from src.search import load_index, search


def test_search_returns_top_k():
    df = load_index()
    results = search("What are Jupyter Notebooks?", df, top_k=5)
    assert len(results) == 5
    assert "similarity" in results.columns
    # Results should be sorted descending by similarity
    scores = results["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_top_result_is_plausible():
    df = load_index()
    results = search("What are Jupyter Notebooks?", df, top_k=1)
    top = results.iloc[0]
    print(f"\nTop match: {top['title']} (score={top['similarity']:.3f})")
    # Loose sanity check — a totally broken embedder would score near 0
    assert top["similarity"] > 0.1