import re

from langchain_core.documents import Document

from prime_land.config import CONFIG

CRAG = CONFIG["crag"]
STOP_WORDS = {
    "the", "and", "for", "are", "what", "which", "how", "can", "you", "have", "with",
    "that", "this", "there", "does", "from", "about", "tell", "want", "show", "any",
}


def format_docs(docs: list[Document]) -> str:
    """Join retrieved documents into one context string with source urls."""
    return "\n\n".join(f"[{doc.metadata['url']}]\n{doc.page_content}" for doc in docs)


def calculate_confidence(docs: list[Document], query: str) -> float:
    """Score retrieval from query keyword coverage and mean similarity, between 0 and 1."""
    if not docs:
        return 0.0
    terms = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2 and word not in STOP_WORDS}
    text = " ".join(doc.page_content.lower() for doc in docs)
    keyword_score = sum(term in text for term in terms) / len(terms) if terms else 0.0
    mean_score = sum(doc.metadata["score"] for doc in docs) / len(docs)
    similarity_score = min(mean_score / CRAG["similarity_scale"], 1.0)
    return CRAG["keyword_weight"] * keyword_score + (1 - CRAG["keyword_weight"]) * similarity_score
