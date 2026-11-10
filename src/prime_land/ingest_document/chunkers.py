import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from prime_land.config import CONFIG

SETTINGS = CONFIG["chunking"]
ENCODING = tiktoken.get_encoding(SETTINGS["encoding"])
METADATA_FIELDS = ["property_id", "property_type", "address", "price"]


def count_tokens(text: str) -> int:
    """Count tokens in a text."""
    return len(ENCODING.encode(text))


def token_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a splitter that measures chunk size in tokens."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=SETTINGS["encoding"], chunk_size=size, chunk_overlap=overlap
    )


def make_chunk(document: dict, text: str, strategy: str, index: int, **extra) -> dict:
    """Build a chunk record that carries the document metadata."""
    metadata = document.get("metadata", {})
    chunk = {
        "url": document["url"],
        "title": document["title"],
        "text": text.strip(),
        "strategy": strategy,
        "chunk_index": index,
        "token_count": count_tokens(text),
    }
    chunk.update({field: metadata.get(field) for field in METADATA_FIELDS})
    chunk.update(extra)
    return chunk


def fixed_chunk(documents: list[dict]) -> list[dict]:
    """Split every document into fixed token windows with overlap."""
    config = SETTINGS["fixed"]
    splitter = token_splitter(config["chunk_size"], config["chunk_overlap"])
    chunks = []
    for document in documents:
        for text in splitter.split_text(document["content"]):
            chunks.append(make_chunk(document, text, "fixed", len(chunks)))
    return chunks


def sliding_chunk(documents: list[dict]) -> list[dict]:
    """Slide a token window over every document using a stride."""
    config = SETTINGS["sliding"]
    chunks = []
    for document in documents:
        tokens = ENCODING.encode(document["content"])
        for start in range(0, len(tokens), config["stride_size"]):
            text = ENCODING.decode(tokens[start : start + config["window_size"]])
            if text.strip():
                chunks.append(make_chunk(document, text, "sliding", len(chunks)))
            if start + config["window_size"] >= len(tokens):
                break
    return chunks


def semantic_chunk(documents: list[dict]) -> list[dict]:
    """Split documents at markdown headings and merge sections that are too small."""
    config = SETTINGS["semantic"]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    size_splitter = token_splitter(config["max_chunk_size"], 0)
    chunks = []
    for document in documents:
        pieces = []
        for section in header_splitter.split_text(document["content"]):
            text = section.page_content.strip()
            if count_tokens(text) > config["max_chunk_size"]:
                pieces.extend(size_splitter.split_text(text))
            elif text:
                pieces.append(text)

        buffer = ""
        for piece in pieces:
            if buffer and (
                count_tokens(buffer) >= config["min_chunk_size"]
                or count_tokens(f"{buffer}\n\n{piece}") > config["max_chunk_size"]
            ):
                chunks.append(make_chunk(document, buffer, "semantic", len(chunks)))
                buffer = piece
            else:
                buffer = f"{buffer}\n\n{piece}" if buffer else piece
        if buffer.strip():
            chunks.append(make_chunk(document, buffer, "semantic", len(chunks)))
    return chunks


def parent_child_chunk(documents: list[dict]) -> list[dict]:
    """Create small child chunks that carry the text of their larger parent."""
    config = SETTINGS["parent_child"]
    parent_splitter = token_splitter(config["parent_size"], 0)
    child_splitter = token_splitter(config["child_size"], config["child_overlap"])
    chunks = []
    for document in documents:
        for parent_index, parent_text in enumerate(parent_splitter.split_text(document["content"])):
            parent_id = f"{document['url']}::{parent_index}"
            for child_text in child_splitter.split_text(parent_text):
                chunks.append(
                    make_chunk(
                        document,
                        child_text,
                        "parent_child",
                        len(chunks),
                        parent_id=parent_id,
                        parent_text=parent_text,
                    )
                )
    return chunks


def late_chunk(documents: list[dict]) -> list[dict]:
    """Split documents into small chunks and record where each one sits in the page text."""
    config = SETTINGS["late"]
    splitter = token_splitter(config["chunk_size"], config["chunk_overlap"])
    chunks = []
    for document in documents:
        content = document["content"]
        position = 0
        for text in splitter.split_text(content):
            start = max(content.find(text, position), position)
            end = min(start + len(text), len(content))
            chunks.append(make_chunk(document, text, "late", len(chunks), char_start=start, char_end=end))
            position = start + 1
    return chunks


CHUNKERS = {
    "semantic": semantic_chunk,
    "fixed": fixed_chunk,
    "sliding": sliding_chunk,
    "parent_child": parent_child_chunk,
    "late": late_chunk,
}


def chunk_documents(documents: list[dict], strategy: str) -> list[dict]:
    """Chunk documents with the named strategy."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy {strategy}, choose from {list(CHUNKERS)}")
    return CHUNKERS[strategy](documents)
