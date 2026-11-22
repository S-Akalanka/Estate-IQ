import streamlit as st
from groq import RateLimitError

from prime_land.chat_service.cag_cache import CAGCache
from prime_land.chat_service.cag_service import CAGService
from prime_land.chat_service.crag_service import CRAGService
from prime_land.chat_service.rag_service import RAGService
from prime_land.config import CONFIG, get_path, load_faqs
from prime_land.ingest_document.indexer import get_client, open_store
from prime_land.providers import get_embeddings, get_llm

MODES = {
    "RAG": "Retrieve chunks and answer.",
    "CRAG": "RAG plus a confidence check that rewrites weak searches.",
    "CAG": "A cache in front of CRAG, so repeated questions skip the LLM.",
}


@st.cache_resource(show_spinner="Loading the models and warming the FAQ cache")
def load_services() -> dict:
    """Load the models and the index once and build the three answer services."""
    strategy = CONFIG["retrieval"]["strategy"]
    embeddings, llm = get_embeddings(), get_llm()
    store = open_store(get_client(), embeddings, strategy)
    rag = RAGService(store, embeddings, llm, strategy=strategy)
    crag = CRAGService(rag, llm)
    cache = CAGCache(embeddings, get_path("cache_dir"), CONFIG["cag"]["similarity_threshold"], CONFIG["cag"]["history_ttl_hours"])
    cag = CAGService(crag, cache)
    cag.warm_faqs(load_faqs())
    return {"RAG": rag, "CRAG": crag, "CAG": cag}


def status_line(result: dict) -> str:
    """Summarise how an answer was produced."""
    parts = []
    if "cache_hit" in result:
        parts.append(f"cache hit from {result['cache_source']} ({result['similarity']:.2f})" if result["cache_hit"] else "cache miss")
    if "confidence_final" in result:
        parts.append(f"confidence {result['confidence_initial']:.2f} to {result['confidence_final']:.2f}")
        parts.append("search corrected" if result["correction_applied"] else "search kept")
    parts.append(f"{result['generation_time']:.1f}s")
    return " | ".join(parts)


def show(message: dict) -> None:
    """Draw one chat message with its sources and status."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown("**Sources**\n\n" + "\n".join(f"- {url}" for url in message["sources"]))
        if message.get("status"):
            st.caption(f"{message['mode']}: {message['status']}")


st.set_page_config(page_title="Prime Lands Assistant")
st.title("Prime Lands Assistant")
st.caption("Ask about Prime Lands lands, houses and apartments.")

mode = st.sidebar.selectbox("Answer mode", list(MODES), index=2)
st.sidebar.caption(MODES[mode])
st.sidebar.subheader("Example questions")
for faq in load_faqs():
    if st.sidebar.button(faq, key=faq):
        st.session_state["pending"] = faq
if st.sidebar.button("Clear chat"):
    st.session_state["messages"] = []

try:
    services = load_services()
except RuntimeError as error:
    st.error(f"Could not open the index. Close any notebook that uses it and reload. {error}")
    st.stop()

messages = st.session_state.setdefault("messages", [])
for message in messages:
    show(message)

question = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if question:
    messages.append({"role": "user", "content": question})
    show(messages[-1])
    try:
        with st.spinner("Thinking"):
            result = services[mode].generate(question)
        reply = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["evidence_urls"],
            "status": status_line(result),
            "mode": mode,
        }
    except RateLimitError:
        reply = {"role": "assistant", "content": "The Groq daily limit is reached. Try again later.", "mode": mode}
    messages.append(reply)
    show(reply)
