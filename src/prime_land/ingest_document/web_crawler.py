import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from collections import deque

class PrimeLandCrawler:
    def __init__(self, base_url:str, max_pages:int):
        self.base_url=base_url
        self.max_pages=max_pages

    async def crawl_async(self):
        queue=deque([self.base_url])
        visited = set()

        async with async_playwright() as p:
            browser=await p.chromium.launch(headless=True)
            page=await browser.new_page()
            page.set_default_timeout(30000)

            while queue and visited < self.max_pages:
                url=queue.popleft()
                if url in visited:
                    continue
                visited.add(url)

                try:
                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )
                    print("crawling: ",url)

                    try:
                        await page.wait_for_selector("body", timeout=10000)
                        await page.wait_for_timeout(1000)

                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(1000)

                    except:
                        await page.wait_for_timeout(5000)

                    html=await page.content()
                    soup=BeautifulSoup(html, 'html.parser')
                    
                except Exception as e:
                    print(e)


    def crawl(self):
        return asyncio.run(self.crawl_async())


if __name__=="__main__":

    BASE_URL = "https://www.primelands.lk"

    START_PATHS = [
        "/en", "/land/en", "/house/en", "/apartment/en",
        "/portfolio-property/en", "/upcoming-projects/en",
        "/about-us/en", "/contact-us/en", "/sell-your-land/en",
        "/services/en", "/testimonial/en", "/careers/en",
        "/news/en", "/online-publications/en", "/blog/en",
        "/virtual-tour/en", "/kyc/en", "/sitemap/en",
        "/privacy-policy/en", "/terms-and-conditions/en", "/disclaimer/en",
        "/land/city/Kadawatha/en", "/land/city/Homagama/en", "/land/city/Kandy/en",
        "/land/city/Anuradhapura/en", "/land/city/Kurunegala/en", "/land/city/Horana/en",
        "/land/city/Malabe/en", "/land/city/Kalutara/en", "/land/city/Gampaha/en",
        "/land/city/Colombo/en",
        "/house/city/Thalawathugoda/en", "/house/city/Kottawa/en", "/house/city/Kadawatha/en",
        "/house/city/Piliyandala/en", "/house/city/Nugegoda/en", "/house/city/Kelaniya/en",
        "/house/city/Kandy/en", "/house/city/Kalutara/en", "/house/city/Malabe/en",
        "/house/city/Nawagamuwa/en",
    ]

    START_URLS = [BASE_URL + path for path in START_PATHS]

    ALLOWED_LANG_SUFFIX = "/en"
    EXCLUDE_LANG_SUFFIXES = ("/sin", "/tam")

    EXCLUDE_PATTERNS = [
        "/public/assets/", "/cdn-cgi/", "/admin", "/login",
        "/ims-policy",
    ]

    crawler=PrimeLandCrawler(BASE_URL,5)
    crawler.crawl()
