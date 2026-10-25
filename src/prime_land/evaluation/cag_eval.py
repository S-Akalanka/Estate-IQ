import json
import random

from prime_land.config import CONFIG, get_path

EVALUATION = CONFIG["evaluation"]
COST = EVALUATION["cost"]


def query_cost() -> float:
    """Estimated dollar cost of one uncached answer."""
    return (
        COST["prompt_tokens"] * COST["input_price_per_million"]
        + COST["completion_tokens"] * COST["output_price_per_million"]
    ) / 1e6


def build_queries(faqs: list[str]) -> list[str]:
    """Create a repeated and paraphrased query stream for the simulation."""
    rng = random.Random(EVALUATION["seed"])
    base = faqs + [item["question"] for item in EVALUATION["queries"]]
    weights = [1 / (rank + 1) for rank in range(len(base))]
    queries = []
    for question in rng.choices(base, weights=weights, k=EVALUATION["cag_queries"]):
        if rng.random() < 0.5:
            question = rng.choice(EVALUATION["paraphrase_prefixes"]) + question[0].lower() + question[1:]
        queries.append(question)
    return queries


def simulate_cag(cag_service, faqs: list[str]) -> dict:
    """Run the query stream through CAG and save hit rate, latency and savings."""
    cag_service.cache.history.clear()
    cag_service.hits = cag_service.misses = cag_service.faq_hits = cag_service.history_hits = 0

    hit_times, miss_times = [], []
    queries = build_queries(faqs)
    for number, query in enumerate(queries, 1):
        result = cag_service.generate(query)
        (hit_times if result["cache_hit"] else miss_times).append(result["generation_time"])
        print(f"simulating {number}/{len(queries)} {'hit' if result['cache_hit'] else 'miss'} {result['generation_time']:.1f}s")

    stats = cag_service.stats()
    avg_hit = sum(hit_times) / len(hit_times) if hit_times else 0.0
    avg_miss = sum(miss_times) / len(miss_times) if miss_times else 0.0
    stats.update(
        {
            "avg_hit_latency_s": round(avg_hit, 3),
            "avg_miss_latency_s": round(avg_miss, 3),
            "latency_improvement_pct": round((1 - avg_hit / avg_miss) * 100, 1) if avg_miss else 0.0,
            "cost_per_uncached_query_usd": round(query_cost(), 6),
            "estimated_savings_usd": round(stats["hits"] * query_cost(), 6),
            "api_calls_avoided": stats["hits"],
        }
    )

    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cag_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
