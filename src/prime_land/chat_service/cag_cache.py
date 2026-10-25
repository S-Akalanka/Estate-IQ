import json
import time
from pathlib import Path

import numpy as np


class CAGCache:
    """Two tier semantic cache with permanent FAQs and history that expires."""

    def __init__(self, embeddings, directory: Path, threshold: float, ttl_hours: float):
        self.embeddings = embeddings
        self.directory = directory
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self.faqs = self.load("faqs.json")
        self.history = self.load("history.json")

    def load(self, filename: str) -> dict:
        """Read one cache tier from disk."""
        path = self.directory / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def save(self) -> None:
        """Write both cache tiers to disk."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for filename, tier in (("faqs.json", self.faqs), ("history.json", self.history)):
            (self.directory / filename).write_text(json.dumps(tier), encoding="utf-8")

    def embed(self, query: str) -> list[float]:
        """Embed a query."""
        return self.embeddings.embed_query(query)

    def find(self, tier: dict, vector: list[float]) -> tuple[dict | None, float]:
        """Return the most similar answered entry in a tier and its similarity."""
        entries = [entry for entry in tier.values() if entry["answer"] is not None]
        if not entries:
            return None, 0.0
        matrix = np.array([entry["embedding"] for entry in entries])
        query = np.array(vector)
        scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        best = int(scores.argmax())
        return entries[best], float(scores[best])

    def prune_history(self) -> None:
        """Drop history entries older than the time to live."""
        now = time.time()
        self.history = {q: e for q, e in self.history.items() if now - e["created_at"] < self.ttl_seconds}

    def get_exact(self, query: str) -> dict | None:
        """Return a cached answer for the same question text without embedding it."""
        self.prune_history()
        key = " ".join(query.lower().split())
        for source, tier in (("faq", self.faqs), ("history", self.history)):
            for entry in tier.values():
                if entry["answer"] is not None and " ".join(entry["query"].lower().split()) == key:
                    return {**entry, "source": source, "similarity": 1.0}
        return None

    def get(self, vector: list[float]) -> dict | None:
        """Look up a query vector in the FAQs first and then in the history."""
        self.prune_history()
        for source, tier in (("faq", self.faqs), ("history", self.history)):
            entry, similarity = self.find(tier, vector)
            if entry and similarity >= self.threshold:
                return {**entry, "source": source, "similarity": similarity}
        return None

    def set_faq(self, query: str, answer: str, evidence_urls: list[str]) -> None:
        """Store the answer for a FAQ question."""
        embedding = self.faqs.get(query, {}).get("embedding") or self.embed(query)
        self.faqs[query] = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "evidence_urls": evidence_urls,
            "created_at": time.time(),
        }
        self.save()

    def add_history(self, query: str, vector: list[float], answer: str, evidence_urls: list[str]) -> None:
        """Store the answer to a user query in the history tier."""
        self.history[query] = {
            "query": query,
            "embedding": vector,
            "answer": answer,
            "evidence_urls": evidence_urls,
            "created_at": time.time(),
        }
        self.save()

    def clear(self) -> None:
        """Remove every cached entry."""
        self.faqs, self.history = {}, {}
        self.save()
