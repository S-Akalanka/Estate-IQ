# Prime Lands Assistant

A personal project that crawls the Prime Lands website (primelands.lk), turns the listings into a searchable knowledge base, and answers questions about them.

I built it to explore how different chunking strategies, caching and retrieval correction change answer quality and cost, using real property pages as the data.

## What it does

1. **Crawls** the site with an async breadth-first crawler that respects a rate limit, and parses each page into markdown plus property metadata (price, address, bedrooms, and so on).
2. **Saves** the pages as one markdown file each and as a single JSONL corpus.
3. **Chunks** the corpus with five strategies: `semantic`, `fixed`, `sliding`, `parent_child` and `late`.
4. **Indexes** each strategy's chunks into a local Qdrant collection, with embeddings from a local sentence transformer model.
5. **Answers questions** with three approaches:
   - **RAG**: retrieve relevant chunks and generate an answer with the LLM.
   - **CAG**: cache answers to common questions, so repeat questions skip the LLM.
   - **CRAG**: score retrieval confidence and correct weak results before answering.
6. **Evaluates** the strategies on precision, recall, answer relevance, latency and cost.

The LLM is served through Groq. Embeddings run locally (nomic-ai/modernbert-embed-base), so indexing needs no API key and has no quota. The first run downloads the model, about 600 MB.

## Project layout

```
config/                 YAML settings for crawling, chunking, retrieval, CAG and evaluation
src/prime_land/
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

## Running it

Run the notebooks in `src/prime_land/notebooks/` in order:

1. `01_crawl_primelands.ipynb`: crawl the site and save the markdown and corpus.
   - Result: 150 pages saved as markdown and as corpus entries.
   - Property pages (34 of them) have most fields filled: price, title and property ID are complete, address is 33/34, agent is 32/34. Bedrooms (21/34), bathrooms (10/34), amenities (18/34) and square feet (5/34) are sparse, so they can't be relied on for filtering.
2. `02_chunk_and_index.ipynb`: chunk the corpus and index it in Qdrant.
   - Result: 77 documents after the exclude patterns are applied. They split into 455 semantic, 185 fixed, 363 sliding, 580 parent-child and 148 late chunks.
   - Indexing reported that every strategy was already complete on this run, so no new points were added and the 900-point budget was left untouched.
   - Retrieval comparison: semantic chunks average 233 tokens and fixed average 608. Sliding averages 486, parent-child 205 and late 696. Index sizes range from 0.45 MB (late) to 1.78 MB (parent-child).
   - Late chunking's average retrieval time is about 20 seconds, far above the others (620 to 740 ms). The log shows rate-limit waits during that cell, so that number probably reflects waiting rather than search speed.
3. `03_intelligence_layers.ipynb`: try RAG, CAG and CRAG on questions.
4. `04_performance_arena.ipynb`: compare the chunking strategies and the caching and correction results, and estimate cost.

Settings such as crawl limits, chunk sizes and the retrieval strategy live in `config/config.yaml`.

## Chat app

After running notebooks 02 and 03, ask questions in a browser:

```bash
uv run streamlit run src/prime_land/app.py
```

Pick the answer mode in the sidebar: RAG, CRAG, or CAG (a cache in front of CRAG). Each answer shows its source pages, whether it came from the cache, the CRAG confidence and the time taken. Close any notebook that uses the index first, because local Qdrant allows one client at a time.

## Notes

- Qdrant runs locally, and its data is kept under `data/qdrant/`.
- The `data/` folder holds generated output and is excluded from version control where it is large.
- This project is not affiliated with Prime Lands. It only reads public pages from their website.
