from langchain_core.documents import Document


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def parent_documents(hits: list[tuple[Document, float]]) -> list[Document]:
    """Swap each child hit for its parent passage and drop duplicates."""
    seen = set()
    documents = []
    for document, score in hits:
        parent_id = document.metadata["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        metadata = {**document.metadata, "score": score}
        documents.append(Document(page_content=metadata.pop("parent_text"), metadata=metadata))
    return documents


def retrieve(store, strategy: str, query: str, k: int) -> list[Document]:
    """Retrieve top k documents, swapping children for parents when the strategy needs it."""
    hits = store.similarity_search_with_score(query, k=k * 3 if strategy == "parent_child" else k)
    if strategy == "parent_child":
        return parent_documents(hits)[:k]
    return [Document(page_content=d.page_content, metadata={**d.metadata, "score": s}) for d, s in hits]
