import json
import re
from pathlib import Path
from urllib.parse import urlparse


def save_markdown(documents: list[dict], directory: Path) -> int:
    """Write one markdown file per document and return how many were written."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(documents):
        slug = urlparse(document["url"]).path.strip("/").replace("/", "_") or "home"
        header = f"# {document['title']}\n\nURL: {document['url']}\n\n---\n\n"
        (directory / f"{index:03d}_{slug}.md").write_text(header + document["content"], encoding="utf-8")
    return len(documents)


def save_jsonl(documents: list[dict], path: Path) -> int:
    """Write the corpus as one json object per line and return the line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")
    return len(documents)


def load_corpus(path: Path, exclude_patterns: list[str]) -> list[dict]:
    """Read the JSONL corpus and drop pages whose url matches an exclude pattern."""
    with open(path, encoding="utf-8") as file:
        documents = [json.loads(line) for line in file]
    return [d for d in documents if not any(re.search(pattern, d["url"]) for pattern in exclude_patterns)]
