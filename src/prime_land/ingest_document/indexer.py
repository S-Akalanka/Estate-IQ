import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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


def index_chunks(client: QdrantClient, embeddings, chunks: list[dict], strategy: str) -> QdrantVectorStore:
    """Create a fresh collection for a strategy and fill it with embedded chunks."""
    name = collection_name(strategy)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=PROVIDER["embedding_dimensions"], distance=Distance.COSINE),
    )
    store = QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)
    documents = [
        Document(page_content=chunk["text"], metadata={k: v for k, v in chunk.items() if k != "text"})
        for chunk in chunks
    ]
    store.add_documents(documents, batch_size=PROVIDER["embedding_batch_size"])
    return store


def open_store(client: QdrantClient, embeddings, strategy: str) -> QdrantVectorStore:
    """Open an existing collection as a vector store."""
    return QdrantVectorStore(client=client, collection_name=collection_name(strategy), embedding=embeddings)
