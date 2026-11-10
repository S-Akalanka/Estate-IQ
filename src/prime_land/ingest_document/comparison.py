import time

import pandas as pd

from prime_land.chat_service.retrieval import retrieve
from prime_land.config import CONFIG, get_path
from prime_land.ingest_document.indexer import collection_name

PROVIDER = CONFIG["provider"]
RETRIEVAL = CONFIG["retrieval"]
QUERIES = [item["question"] for item in CONFIG["evaluation"]["queries"]]


def compare_strategies(client, stores: dict, chunk_sets: dict) -> pd.DataFrame:
    """Build the table of chunk count, size, index size and retrieval time per strategy."""
    rows = []
    for strategy, chunks in chunk_sets.items():
        vectors = client.count(collection_name(strategy)).count
        elapsed = []
        for query in QUERIES:
            start = time.perf_counter()
            retrieve(stores[strategy], strategy, query, RETRIEVAL["top_k"])
            elapsed.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "strategy": strategy,
                "chunk_count": len(chunks),
                "avg_tokens": round(sum(c["token_count"] for c in chunks) / len(chunks), 1),
                "index_size_mb": round(vectors * PROVIDER["embedding_dimensions"] * 4 / 1e6, 2),
                "avg_retrieval_ms": round(sum(elapsed) / len(elapsed), 1),
            }
        )
    table = pd.DataFrame(rows)
    path = get_path("results_dir") / "index_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
