import time
from collections import deque

from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from prime_land.config import CONFIG, get_api_key
from prime_land.ingest_document.chunkers import count_tokens

PROVIDER = CONFIG["provider"]
WINDOW_SECONDS = 60


class ThrottledEmbeddings(Embeddings):
    """Embedding wrapper that stays under a tokens per minute limit and retries on 429 errors."""

    def __init__(self, inner: Embeddings):
        self.inner = inner
        self.limit = PROVIDER["embedding_tokens_per_minute"]
        self.window = deque()

    def wait_for(self, tokens: int) -> None:
        """Block until the token budget of the last minute has room for tokens."""
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0][0] >= WINDOW_SECONDS:
                self.window.popleft()
            used = sum(count for _, count in self.window)
            if not self.window or used + tokens <= self.limit:
                break
            pause = self.window[0][0] + WINDOW_SECONDS - now + 0.1
            print(f"token budget used {used}, waiting {pause:.0f}s")
            time.sleep(pause)
        self.window.append((time.monotonic(), tokens))

    def call(self, function, texts: list[str]):
        """Run one embedding request within the budget, retrying when rate limited."""
        tokens = sum(count_tokens(text) for text in texts)
        for attempt in range(PROVIDER["max_retries"] + 1):
            self.wait_for(tokens)
            try:
                return function(texts)
            except Exception as error:
                limited = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
                if not limited or attempt == PROVIDER["max_retries"]:
                    raise
                print(f"rate limited, waiting {PROVIDER['retry_wait_seconds']}s")
                time.sleep(PROVIDER["retry_wait_seconds"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in groups that each fit the token budget."""
        vectors, group, group_tokens = [], [], 0
        for text in texts:
            tokens = count_tokens(text)
            if group and group_tokens + tokens > self.limit:
                vectors += self.call(self.inner.embed_documents, group)
                group, group_tokens = [], 0
            group.append(text)
            group_tokens += tokens
        if group:
            vectors += self.call(self.inner.embed_documents, group)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self.call(lambda texts: self.inner.embed_query(texts[0]), [text])


def get_embeddings() -> ThrottledEmbeddings:
    """Create the Gemini embedding model with rate limiting."""
    inner = GoogleGenerativeAIEmbeddings(
        model=PROVIDER["embedding_model"],
        google_api_key=get_api_key(),
        output_dimensionality=PROVIDER["embedding_dimensions"],
    )
    return ThrottledEmbeddings(inner)


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model with rate limiting."""
    limiter = InMemoryRateLimiter(
        requests_per_second=PROVIDER["chat_requests_per_minute"] / 60, max_bucket_size=1
    )
    return ChatGoogleGenerativeAI(
        model=PROVIDER["chat_model"],
        temperature=PROVIDER["temperature"],
        max_output_tokens=PROVIDER["max_tokens"],
        max_retries=PROVIDER["max_retries"],
        rate_limiter=limiter,
        google_api_key=get_api_key(),
    )
