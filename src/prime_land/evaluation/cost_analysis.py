import json

from prime_land.config import CONFIG, get_path
from prime_land.evaluation.cag_eval import query_cost

COST = CONFIG["evaluation"]["cost"]
DIMENSIONS = CONFIG["provider"]["embedding_dimensions"]


def analyze_cost(cag_stats: dict, vector_count: int) -> dict:
    """Estimate monthly API and storage cost at scale and the return against human agents."""
    monthly_queries = COST["daily_users"] * COST["queries_per_user"] * COST["days_per_month"]
    embedding_cost = 30 * COST["embedding_price_per_million"] / 1e6
    hit_rate = cag_stats["hit_rate"]

    api_without_cache = monthly_queries * (query_cost() + embedding_cost)
    api_with_cache = monthly_queries * (1 - hit_rate) * query_cost() + monthly_queries * embedding_cost
    storage_gb = vector_count * DIMENSIONS * 4 / 1e9
    storage = storage_gb * COST["qdrant_price_per_gb_month"]
    system_cost = api_with_cache + storage
    human_cost = monthly_queries * COST["human_agent_cost_per_query"]

    report = {
        "monthly_queries": monthly_queries,
        "cache_hit_rate": hit_rate,
        "api_cost_without_cache_usd": round(api_without_cache, 2),
        "api_cost_with_cache_usd": round(api_with_cache, 2),
        "storage_cost_usd": round(storage, 4),
        "monthly_system_cost_usd": round(system_cost, 2),
        "monthly_cache_savings_usd": round(api_without_cache - api_with_cache, 2),
        "human_agent_cost_usd": round(human_cost, 2),
        "roi_percent": round((human_cost - system_cost) / system_cost * 100, 1),
    }

    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cost_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
