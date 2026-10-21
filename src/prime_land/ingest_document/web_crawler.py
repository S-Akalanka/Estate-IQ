import asyncio
import re
from collections import deque
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from prime_land.ingest_document.page_parser import build_document

MEDIA_PATTERN = re.compile(r"\.(jpg|jpeg|png|gif|svg|webp|pdf|zip|mp4)$", re.I)


class PrimeLandCrawler:
    """Async breadth first crawler for JavaScript rendered Prime Lands pages."""

    def __init__(
        self,
        base_url: str,
        language_suffix: str,
        exclude_patterns: list[str],
        max_depth: int,
        max_pages: int,
        delay_seconds: float,
        page_timeout_ms: int,
        settle_ms: int,
        min_content_chars: int,
    ):
        self.base_url = base_url
        self.language_suffix = language_suffix
        self.exclude_patterns = exclude_patterns
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay_seconds = delay_seconds
        self.page_timeout_ms = page_timeout_ms
        self.settle_ms = settle_ms
        self.min_content_chars = min_content_chars
        self.documents: list[dict] = []
        self.failed: list[str] = []

    def normalize(self, url: str) -> str:
        """Resolve a url against the base and drop query and fragment."""
        absolute = urljoin(self.base_url, url)
        return urlparse(absolute)._replace(query="", fragment="").geturl()

    def should_crawl(self, url: str) -> bool:
        """Return True for english pages on the site that are not excluded."""
        if not url.startswith(self.base_url):
            return False
        if not url.endswith(self.language_suffix):
            return False
        if MEDIA_PATTERN.search(url):
            return False
        return not any(pattern in url for pattern in self.exclude_patterns)

    def extract_links(self, soup: BeautifulSoup, url: str) -> list[str]:
        """Collect crawlable links found on a page."""
        links = {self.normalize(urljoin(url, a["href"])) for a in soup.find_all("a", href=True)}
        return [link for link in links if self.should_crawl(link)]

    async def fetch(self, page, url: str) -> str | None:
        """Load a page, let it render, and return its html or None on failure."""
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector("body")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(self.settle_ms)
            return await page.content()
        except Exception as error:
            print(f"failed {url}: {str(error)[:100]}")
            self.failed.append(url)
            return None

    async def crawl_async(self, start_urls: list[str]) -> list[dict]:
        """Crawl from the start urls up to max_depth and max_pages."""
        queue = deque()
        seen = set()

        def enqueue(url: str, depth: int) -> None:
            if url not in seen and self.should_crawl(url):
                seen.add(url)
                queue.append((url, depth))

        for start_url in start_urls:
            enqueue(self.normalize(start_url), 0)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                page.set_default_timeout(self.page_timeout_ms)

                while queue and len(self.documents) < self.max_pages:
                    url, depth = queue.popleft()
                    html = await self.fetch(page, url)
                    if html is None:
                        continue

                    soup = BeautifulSoup(html, "html.parser")
                    links = self.extract_links(soup, url)
                    document = build_document(soup, url, depth)

                    if len(document["content"]) >= self.min_content_chars:
                        self.documents.append(document)
                        print(f"[{depth}] saved {len(self.documents)}: {url}")

                    if depth < self.max_depth:
                        for link in links:
                            enqueue(link, depth + 1)

                    await asyncio.sleep(self.delay_seconds)
            finally:
                await browser.close()

        return self.documents
