import json
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from prime_land.config import CONFIG, get_path

PROVIDER = CONFIG["provider"]
RETRIEVAL = CONFIG["retrieval"]


def collection_name(strategy: str) -> str:
    """Return the Qdrant collection name for a strategy."""
    return f"{RETRIEVAL['collection_prefix']}_{strategy}"


def get_client() -> QdrantClient:
    """Open the persistent local Qdrant store."""
    path = get_path("vector_dir")
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def save_chunks(chunks: list[dict], strategy: str) -> Path:
    """Write chunks of one strategy to a JSONL file."""
    path = get_path("chunks_dir") / f"chunks_{strategy}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return path


def load_chunks(strategy: str) -> list[dict]:
    """Read chunks of one strategy from its JSONL file."""
    path = get_path("chunks_dir") / f"chunks_{strategy}.jsonl"
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def late_vectors(embeddings, chunks: list[dict], documents: list[dict]) -> list[list[float]]:
    """Embed each page once and average the token vectors of its chunks."""
    content = {document["url"]: document["content"] for document in documents}
    vectors = []
    for url in dict.fromkeys(chunk["url"] for chunk in chunks):
        spans = [(chunk["char_start"], chunk["char_end"]) for chunk in chunks if chunk["url"] == url]
        vectors += embeddings.embed_late(content[url], spans)
    return vectors


def index_chunks(client: QdrantClient, embeddings, chunks: list[dict], strategy: str, documents: list[dict]) -> QdrantVectorStore:
    """Rebuild a strategy collection from its chunks."""
    name = collection_name(strategy)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=PROVIDER["embedding_dimensions"], distance=Distance.COSINE),
    )
    if strategy == "late":
        vectors = late_vectors(embeddings, chunks, documents)
    else:
        vectors = embeddings.embed_documents([chunk["text"] for chunk in chunks])

    points = [
        PointStruct(
            id=index,
            vector=vector,
            payload={"page_content": chunk["text"], "metadata": {k: v for k, v in chunk.items() if k != "text"}},
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    for start in range(0, len(points), 256):
        client.upsert(collection_name=name, points=points[start : start + 256])
    return open_store(client, embeddings, strategy)


def open_store(client: QdrantClient, embeddings, strategy: str) -> QdrantVectorStore:
    """Open an existing collection as a vector store."""
    return QdrantVectorStore(client=client, collection_name=collection_name(strategy), embedding=embeddings)
