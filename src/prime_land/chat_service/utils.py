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
    limit = CONFIG["provider"]["context_chars_per_doc"]
    return "\n\n".join(f"[{doc.metadata['url']}]\n{doc.page_content[:limit]}" for doc in docs)


def confidence_parts(docs: list[Document], query: str) -> tuple[float, float]:
    """Return keyword coverage and scaled mean similarity for retrieved documents."""
    if not docs:
        return 0.0, 0.0
    terms = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2 and word not in STOP_WORDS}
    words = set(re.findall(r"[a-z0-9]+", " ".join(doc.page_content.lower() for doc in docs)))
    keyword_score = len(terms & words) / len(terms) if terms else 0.0
    mean_score = sum(doc.metadata["score"] for doc in docs) / len(docs)
    span = CRAG["similarity_ceiling"] - CRAG["similarity_floor"]
    return keyword_score, min(max((mean_score - CRAG["similarity_floor"]) / span, 0.0), 1.0)


def calculate_confidence(docs: list[Document], query: str) -> float:
    """Score retrieval from query keyword coverage and mean similarity, between 0 and 1."""
    keyword_score, similarity_score = confidence_parts(docs, query)
    return CRAG["keyword_weight"] * keyword_score + (1 - CRAG["keyword_weight"]) * similarity_score
