import streamlit as st
from groq import RateLimitError

from prime_land.chat_service.cag_cache import CAGCache
from prime_land.chat_service.cag_service import CAGService
from prime_land.chat_service.crag_service import CRAGService
from prime_land.chat_service.rag_service import RAGService
from prime_land.config import CONFIG, get_path, load_faqs
from prime_land.ingest_document.indexer import get_client, open_store
from prime_land.ingest_document.writers import load_corpus
from prime_land.providers import get_embeddings, get_llm

LISTING_IDS = {"ongoing", "completed", "upcoming"}
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


@st.cache_data
def load_properties() -> dict:
    """Map each property page url to its metadata."""
    documents = load_corpus(get_path("corpus_file"), CONFIG["corpus"]["exclude_patterns"])
    return {
        document["url"]: document["metadata"]
        for document in documents
        if document["page_type"] == "property" and document["metadata"]["property_id"].lower() not in LISTING_IDS
    }


def price_text(meta: dict) -> str | None:
    """Format the listed price, for example From LKR 1,050,000 per perch."""
    if not meta.get("price"):
        return None
    if not meta.get("price_unit"):
        return f"LKR {meta['price']:,}"
    return f"From LKR {meta['price']:,} {meta['price_unit'].lower().replace(' upwards', '')}"


def property_card(url: str, meta: dict) -> None:
    """Draw one property as a small card."""
    details = [
        (meta.get("property_type") or "").title(),
        (meta.get("status") or "").title(),
        price_text(meta),
        (meta.get("address") or "").title(),
        f"Hotline {meta['agent']}" if meta.get("agent") else None,
    ]
    with st.container(border=True):
        st.markdown(f"**[{meta['title'].title()}]({url})**")
        st.caption(" | ".join(part for part in details if part))


def show(message: dict) -> None:
    """Draw one chat message with its property cards, other sources and status."""
    properties = load_properties()
    sources = message.get("sources", [])
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for url in sources:
            if url in properties:
                property_card(url, properties[url])
        pages = [url for url in sources if url not in properties]
        if pages:
            st.markdown("**Other pages**\n\n" + "\n".join(f"- {url}" for url in pages))
        if message.get("status"):
            st.caption(f"{message['mode']}: {message['status']}")


st.set_page_config(page_title="Estate IQ")
st.title("Estate IQ")
st.caption("Ask about lands, houses and apartments on the Prime Lands website.")

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
