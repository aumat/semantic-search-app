"""
Streamlit UI for semantic video search.
Run with: streamlit run app.py
"""
import streamlit as st

from src.search import load_index, search
from config import TOP_K

st.set_page_config(page_title="AI Show Semantic Search", layout="centered")

st.title("🔍 Microsoft AI Show — Semantic Search")
st.caption(
    "Search 1,400+ video segments by meaning, not just keywords. "
    "Powered by local sentence embeddings — no external API calls."
)


@st.cache_resource
def get_index():
    """
    Cached so the ~1409-row index (and the embedding model behind it) is
    loaded ONCE per app session, not re-loaded on every keystroke/rerun.
    Streamlit reruns the whole script top-to-bottom on every interaction,
    so this decorator is what keeps the app fast.
    """
    return load_index()


df = get_index()

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What are Jupyter Notebooks?",
)

if query:
    with st.spinner("Searching..."):
        results = search(query, df, top_k=TOP_K)

    st.subheader(f"Top {len(results)} results")

    for _, row in results.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['title']}")
            st.write(row["summary"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Similarity", f"{row['similarity']:.3f}")
            col2.write(f"**Timestamp:** {row['start']}")
            col3.markdown(f"[▶ Watch segment]({row['youtube_url']})")
            st.caption(f"Speaker(s): {row['speaker']}")
else:
    st.info("Enter a question above to search the video library.")