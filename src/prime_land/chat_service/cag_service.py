import time

from prime_land.chat_service.cag_cache import CAGCache
from prime_land.chat_service.rag_service import RAGService
from prime_land.providers import limiter_wait


class CAGService:
    """RAG with a semantic cache in front of it and hit statistics."""

    def __init__(self, rag_service: RAGService, cache: CAGCache):
        self.rag_service = rag_service
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self.faq_hits = 0
        self.history_hits = 0

    def warm_faqs(self, questions: list[str]) -> int:
        """Generate and store answers for FAQ questions that have none yet."""
        warmed = 0
        for number, question in enumerate(questions, 1):
            entry = self.cache.faqs.get(question)
            if entry and entry["answer"] is not None:
                continue
            print(f"warming FAQ {number}/{len(questions)}: {question}")
            result = self.rag_service.generate(question)
            self.cache.set_faq(question, result["answer"], result["evidence_urls"])
            warmed += 1
        return warmed

    def generate(self, query: str) -> dict:
        """Answer from the cache when a similar query exists, otherwise run RAG and cache it."""
        start, waited = time.perf_counter(), limiter_wait()
        cached = self.cache.get_exact(query)
        vector = None
        if cached is None:
            vector = self.cache.embed(query)
            cached = self.cache.get(vector)

        if cached:
            self.hits += 1
            if cached["source"] == "faq":
                self.faq_hits += 1
            else:
                self.history_hits += 1
            return {
                "answer": cached["answer"],
                "evidence_urls": cached["evidence_urls"],
                "cache_hit": True,
                "cache_source": cached["source"],
                "similarity": cached["similarity"],
                "generation_time": time.perf_counter() - start,
            }

        self.misses += 1
        result = self.rag_service.generate(query)
        self.cache.add_history(query, vector, result["answer"], result["evidence_urls"])
        return {
            "answer": result["answer"],
            "evidence_urls": result["evidence_urls"],
            "cache_hit": False,
            "cache_source": None,
            "similarity": 0.0,
            "generation_time": time.perf_counter() - start - (limiter_wait() - waited),
        }

    def stats(self) -> dict:
        """Return cache hit statistics."""
        total = self.hits + self.misses
        return {
            "total_queries": total,
            "hits": self.hits,
            "misses": self.misses,
            "faq_hits": self.faq_hits,
            "history_hits": self.history_hits,
            "hit_rate": self.hits / total if total else 0.0,
        }
