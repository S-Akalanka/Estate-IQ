import time

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from prime_land.config import CONFIG, get_groq_api_key

PROVIDER = CONFIG["provider"]


class LocalEmbeddings(Embeddings):
    """Local sentence transformer that adds the task prefixes and normalizes vectors."""

    def __init__(self):
        self.model = SentenceTransformer(PROVIDER["embedding_model"])
        self.model.max_seq_length = PROVIDER["embedding_max_tokens"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as documents."""
        prefixed = [PROVIDER["document_prefix"] + text for text in texts]
        vectors = self.model.encode(prefixed, batch_size=PROVIDER["embedding_batch_size"], normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self.model.encode(PROVIDER["query_prefix"] + text, normalize_embeddings=True).tolist()

    def embed_late(self, content: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        """Embed a whole text once, then average the token vectors inside each character span."""
        prefix = PROVIDER["document_prefix"]
        vectors, index = [], 0
        while index < len(spans):
            first = index
            origin = spans[first][0]
            while index < len(spans) and (index == first or spans[index][1] - origin <= PROVIDER["late_window_chars"]):
                index += 1
            group = spans[first:index]
            text = prefix + content[origin : group[-1][1]]
            tokens = self.model.encode(text, output_value="token_embeddings")
            offsets = self.model.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.model.max_seq_length)["offset_mapping"]
            for start, end in group:
                low, high = len(prefix) + start - origin, len(prefix) + end - origin
                inside = [i for i, (a, b) in enumerate(offsets) if b > a and a >= low and b <= high]
                vector = np.asarray(tokens[inside] if inside else tokens).mean(axis=0)
                vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors


class TimedLimiter(InMemoryRateLimiter):
    """Rate limiter that adds up how long it made callers wait."""

    waited = 0.0

    def acquire(self, *, blocking: bool = True) -> bool:
        """Take a token and record the time spent waiting for it."""
        start = time.perf_counter()
        allowed = super().acquire(blocking=blocking)
        TimedLimiter.waited += time.perf_counter() - start
        return allowed


def limiter_wait() -> float:
    """Return the total seconds spent waiting on the chat rate limiter so far."""
    return TimedLimiter.waited


def get_embeddings() -> LocalEmbeddings:
    """Create the local embedding model."""
    return LocalEmbeddings()


def get_llm() -> ChatGroq:
    """Create the Groq chat model with rate limiting."""
    limiter = TimedLimiter(requests_per_second=PROVIDER["chat_requests_per_minute"] / 60, max_bucket_size=1)
    return ChatGroq(
        model=PROVIDER["chat_model"],
        temperature=PROVIDER["temperature"],
        max_tokens=PROVIDER["max_tokens"],
        reasoning_effort=PROVIDER["reasoning_effort"],
        max_retries=PROVIDER["max_retries"],
        rate_limiter=limiter,
        groq_api_key=get_groq_api_key(),
    )
