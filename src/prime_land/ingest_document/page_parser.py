import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify

NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "form"]
PROPERTY_TYPES = {"land", "house", "apartment"}
LISTING_SEGMENTS = {"city", "district"}
SECTION_STOPS = {
    "location",
    "location highlights",
    "payment plan",
    "block plan",
    "road map",
    "videos",
    "floor plans",
    "inquire",
    "hotline numbers",
    "amenities",
    "facilities",
    "we also recommend",
}
STATUSES = {"LIVE", "SOLD OUT", "UPCOMING", "COMPLETED", "ONGOING"}
MAX_AMENITY_LENGTH = 80
RECOMMEND_LINE = r"\W*(?:we also )?recommend\W*"


def property_type_of(url: str) -> str | None:
    """Return land, house or apartment for a property detail url, else None."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) == 3 and parts[0] in PROPERTY_TYPES and parts[1] not in LISTING_SEGMENTS:
        return parts[0]
    return None


def value_after(lines: list[str], label: str) -> str | None:
    """Return the line that follows the first line equal to label."""
    for index, line in enumerate(lines):
        if line.lower() == label.lower() and index + 1 < len(lines):
            return lines[index + 1]
    return None


def section_items(lines: list[str], label: str) -> list[str]:
    """Return the short lines listed under a section heading."""
    items = []
    for index, line in enumerate(lines):
        if line.lower() != label.lower():
            continue
        for item in lines[index + 1 :]:
            if item.lower() in SECTION_STOPS or len(item) > MAX_AMENITY_LENGTH:
                break
            items.append(item)
        break
    return items


def first_number(pattern: str, text: str) -> float | None:
    """Return the first number captured by pattern, or None."""
    match = re.search(pattern, text, re.I)
    return float(match.group(1).replace(",", "")) if match else None


def extract_property(lines: list[str], url: str, title: str) -> dict:
    """Build the metadata record for a property detail page."""
    text = "\n".join(lines)
    price = re.search(r"([\d,]{4,})\s*LKR", text)
    price_unit = next((line for line in lines if line.upper().startswith("PER ")), None)
    bedrooms = first_number(r"(\d+)\s*(?:bed(?:room)?s?)\b", text)
    bathrooms = first_number(r"(\d+)\s*(?:bath(?:room)?s?)\b", text)
    sqft = first_number(r"([\d,\.]+)\s*(?:sq\.?\s*ft|sqft|square feet)", text)
    amenities = section_items(lines, "Amenities") or section_items(lines, "Facilities")

    return {
        "property_id": urlparse(url).path.strip("/").split("/")[1],
        "title": title,
        "property_type": property_type_of(url),
        "address": value_after(lines, "Location"),
        "status": next((line for line in lines if line.upper() in STATUSES), None),
        "price": int(price.group(1).replace(",", "")) if price else None,
        "price_unit": price_unit,
        "bedrooms": int(bedrooms) if bedrooms is not None else None,
        "bathrooms": int(bathrooms) if bathrooms is not None else None,
        "sqft": sqft,
        "amenities": amenities,
        "agent": value_after(lines, "Hotline Numbers"),
    }


def build_document(soup: BeautifulSoup, url: str, depth: int) -> dict:
    """Turn a rendered page into a document with markdown content and metadata."""
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    heading = soup.find("h1")
    page_title = soup.title.string if soup.title and soup.title.string else url
    title = heading.get_text(strip=True) if heading else page_title.strip()
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3", "h4"])]
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    cut = next((i for i, line in enumerate(lines) if re.fullmatch(RECOMMEND_LINE, line, re.I)), len(lines))
    lines = lines[:cut]

    content = markdownify(str(soup.body or soup), heading_style="ATX", strip=["a", "img"])
    content = re.split(f"^{RECOMMEND_LINE}$", content, flags=re.I | re.M)[0]
    first_heading = re.search(r"^#", content, re.M)
    content = content[first_heading.start() :] if first_heading else content
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    is_property = property_type_of(url) is not None
    return {
        "url": url,
        "title": title,
        "headings": headings,
        "content": content,
        "depth_level": depth,
        "page_type": "property" if is_property else "page",
        "metadata": extract_property(lines, url, title) if is_property else {},
    }
