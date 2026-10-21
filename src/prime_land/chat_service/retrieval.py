from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prime_land.config import CONFIG

SETTINGS = CONFIG["chunking"]


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


def late_documents(hits: list[tuple[Document, float]], embeddings, query: str) -> list[Document]:
    """Split each base passage and keep the piece closest to the query."""
    config = SETTINGS["late"]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=SETTINGS["encoding"],
        chunk_size=config["split_size"],
        chunk_overlap=config["split_overlap"],
    )
    query_vector = embeddings.embed_query(query)
    documents = []
    for document, score in hits:
        pieces = splitter.split_text(document.page_content)
        vectors = embeddings.embed_documents(pieces)
        best = max(range(len(pieces)), key=lambda i: cosine(query_vector, vectors[i]))
        documents.append(Document(page_content=pieces[best], metadata={**document.metadata, "score": score}))
    return documents


def retrieve(store, embeddings, strategy: str, query: str, k: int) -> list[Document]:
    """Retrieve top k documents, applying the strategy specific retrieval step."""
    hits = store.similarity_search_with_score(query, k=k * 3 if strategy == "parent_child" else k)
    if strategy == "parent_child":
        return parent_documents(hits)[:k]
    if strategy == "late":
        return late_documents(hits, embeddings, query)
    return [Document(page_content=d.page_content, metadata={**d.metadata, "score": s}) for d, s in hits]
