import json

import pandas as pd
from groq import RateLimitError

from prime_land.chat_service.rag_service import RAGService
from prime_land.chat_service.retrieval import cosine
from prime_land.config import CONFIG, get_path

RETRIEVAL = CONFIG["retrieval"]
QUERIES = CONFIG["evaluation"]["queries"]


def precision_at_k(texts: list[str], keywords: list[str]) -> float:
    """Share of retrieved texts that contain at least one expected keyword."""
    relevant = sum(any(keyword in text for keyword in keywords) for text in texts)
    return relevant / len(texts) if texts else 0.0


def recall_at_k(texts: list[str], keywords: list[str]) -> float:
    """Share of expected keywords that appear in at least one retrieved text."""
    found = sum(any(keyword in text for text in texts) for keyword in keywords)
    return found / len(keywords)


def evaluate_chunking(stores: dict, embeddings, llm) -> pd.DataFrame:
    """Run every query on every strategy and save per query and summary tables."""
    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    progress = results_dir / "chunking_progress.jsonl"
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()] if progress.exists() else []
    done = {(row["strategy"], row["question"]) for row in rows}
    if rows:
        print(f"resuming from {len(rows)} saved rows")
    total = len(stores) * len(QUERIES)
    try:
        for strategy, store in stores.items():
            service = RAGService(store, embeddings, llm, strategy=strategy, k=RETRIEVAL["top_k"])
            for item in QUERIES:
                if (strategy, item["question"]) in done:
                    continue
                print(f"evaluating {len(rows) + 1}/{total} [{strategy}]: {item['question']}")
                result = service.generate(item["question"])
                texts = [doc.page_content.lower() for doc in result["evidence"]]
                keywords = [keyword.lower() for keyword in item["keywords"]]
                query_vector = embeddings.embed_query(item["question"])
                answer_vector = embeddings.embed_query(result["answer"])
                row = {
                    "strategy": strategy,
                    "question": item["question"],
                    "precision_at_5": precision_at_k(texts, keywords),
                    "recall_at_5": recall_at_k(texts, keywords),
                    "answer_relevance": cosine(query_vector, answer_vector),
                    "latency_s": result["generation_time"],
                }
                rows.append(row)
                with open(progress, "a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
        progress.unlink(missing_ok=True)
    except RateLimitError:
        print(f"Groq limit reached, saving {len(rows)} of {total} rows, rerun to continue")

    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    detail.to_csv(results_dir / "chunking_comparison_detail.csv", index=False)

    summary = detail.drop(columns="question").groupby("strategy").mean().round(3).reset_index()
    summary["score"] = (summary["precision_at_5"] + summary["recall_at_5"] + summary["answer_relevance"]).round(3)
    summary = summary.sort_values("score", ascending=False)
    summary.to_csv(results_dir / "chunking_comparison.csv", index=False)
    return summary
