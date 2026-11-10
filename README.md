# Prime Lands Assistant

A personal project that crawls the Prime Lands website (primelands.lk), turns the listings into a searchable knowledge base, and answers questions about them.

I built it to explore how different chunking strategies, caching and retrieval correction change answer quality and cost, using real property pages as the data.

## What it does

1. **Crawls** the site with an async breadth-first crawler that respects a rate limit, and parses each page into markdown plus property metadata (price, address, bedrooms, and so on).
2. **Saves** the pages as one markdown file each and as a single JSONL corpus.
3. **Chunks** the corpus with five strategies: `semantic`, `fixed`, `sliding`, `parent_child` and `late`.
4. **Indexes** each strategy's chunks into a local Qdrant collection, with embeddings from Google Gemini.
5. **Answers questions** with three approaches:
   - **RAG**: retrieve relevant chunks and generate an answer with the LLM.
   - **CAG**: cache answers to common questions, so repeat questions skip the LLM.
   - **CRAG**: score retrieval confidence and correct weak results before answering.
6. **Evaluates** the strategies on precision, recall, answer relevance, latency and cost.

The LLM is served through Groq.

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
cp .env.example .env   # then add your GOOGLE_API_KEY and GROQ_API_KEY
```

## Running it

Run the notebooks in `src/prime_land/notebooks/` in order:

1. `01_crawl_primelands.ipynb`: crawl the site and save the markdown and corpus.
2. `02_chunk_and_index.ipynb`: chunk the corpus and index it in Qdrant.
3. `03_intelligence_layers.ipynb`: try RAG, CAG and CRAG on questions.
4. `04_performance_arena.ipynb`: compare the chunking strategies and the caching and correction results, and estimate cost.

Settings such as crawl limits, chunk sizes and the retrieval strategy live in `config/config.yaml`.

## Notes

- Qdrant runs locally, and its data is kept under `data/qdrant/`.
- The `data/` folder holds generated output and is excluded from version control where it is large.
- This project is not affiliated with Prime Lands. It only reads public pages from their website.
