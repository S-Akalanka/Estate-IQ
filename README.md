# Estate IQ

An AI assistant for real estate websites. It reads the pages of a property site, currently primelands.lk, turns the listings into a searchable knowledge base, and answers visitor questions about lands, houses and apartments with links to the source pages.

It also measures how chunking strategies, caching and retrieval correction change answer quality, speed and cost, so the setup behind the assistant is chosen from evidence.

## What it does

1. **Crawls** the site with an async breadth first crawler that respects a rate limit, and parses each page into markdown plus property metadata (price, address, status, and so on).
2. **Saves** the pages as one markdown file each and as a single JSONL corpus.
3. **Chunks** the corpus with five strategies: `semantic`, `fixed`, `sliding`, `parent_child` and `late`.
4. **Indexes** each strategy's chunks into a local Qdrant collection, with embeddings from a local sentence transformer model.
5. **Answers questions** with three approaches:
   - **RAG**: retrieve relevant chunks and generate an answer with the LLM.
   - **CAG**: cache answers to common questions, so repeat questions skip the LLM.
   - **CRAG**: score retrieval confidence and correct weak results before answering.
6. **Evaluates** the strategies on precision, recall, answer relevance, latency and cost.

The LLM is served through Groq. Embeddings run locally (nomic-ai/modernbert-embed-base), so indexing needs no API key and has no quota. The first run downloads the model, about 600 MB.

## Chat app

After running notebooks 02 and 03, ask questions in a browser:

```bash
uv run streamlit run src/prime_land/app.py
```

Pick the answer mode in the sidebar: RAG, CRAG, or CAG (a cache in front of CRAG). Each answer shows property cards and source pages, whether it came from the cache, the CRAG confidence and the time taken. Close any notebook that uses the index first, because local Qdrant allows one client at a time.

## Project layout

```
config/                 YAML settings for crawling, chunking, retrieval, CAG and evaluation
src/prime_land/
  app.py                the Streamlit chat app
  ingest_document/      crawler, page parser, writers, chunkers, indexer, comparison
  chat_service/         RAG, CAG, CRAG, retrieval and prompts
  evaluation/           chunking, CAG, CRAG and cost analysis
  providers.py          embedding model, LLM and rate limiting
  notebooks/            the pipeline, run in order (01 to 04)
data/                   crawled pages, corpus, chunks and results
```

## Setup

Requires Python 3.14 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your GROQ_API_KEY
```

## Running the pipeline

Run the notebooks in `src/prime_land/notebooks/` in order:

1. `01_crawl_primelands.ipynb`: crawl the site and save the markdown and corpus.
   - Result: 150 pages saved as markdown and as corpus entries.
   - Property pages (34 of them) have price, title, property ID and status on every page, with address and agent on 33. Amenities (18/34), bedrooms (5/34), square feet (5/34) and bathrooms (1/34) are sparse, so they can't be relied on for filtering.
2. `02_chunk_and_index.ipynb`: chunk the corpus and index it in Qdrant.
   - Result: 77 documents after the exclude patterns are applied. They split into 377 semantic, 161 fixed, 298 sliding, 490 parent child and 391 late chunks.
   - Average chunk size is 237 tokens for semantic, 583 for fixed, 481 for sliding, 203 for parent child and 249 for late. Index sizes range from 1.06 MB (fixed) to 3.29 MB (parent child).
   - Average retrieval time is 100 to 150 ms for every strategy.
3. `03_intelligence_layers.ipynb`: try RAG, CAG and CRAG on questions.
4. `04_performance_arena.ipynb`: compare the chunking strategies and the caching and correction results, and estimate cost. The Groq free tier has a daily token limit, so this notebook saves its progress and can be rerun the next day to continue.

Settings such as crawl limits, chunk sizes and the retrieval strategy live in `config/config.yaml`.

## Notes

- Qdrant runs locally, and its data is kept under `data/qdrant/`.
- The `data/` folder holds generated output and is excluded from version control where it is large.
