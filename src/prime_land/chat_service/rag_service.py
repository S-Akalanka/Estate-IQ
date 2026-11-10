import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from prime_land.chat_service.prompts import RAG_TEMPLATE
from prime_land.chat_service.retrieval import retrieve
from prime_land.chat_service.utils import format_docs
from prime_land.config import CONFIG
from prime_land.providers import limiter_wait

RETRIEVAL = CONFIG["retrieval"]


class RAGService:
    """Answer questions with an LCEL chain over a Qdrant retriever."""

    def __init__(self, store, embeddings, llm, strategy: str = RETRIEVAL["strategy"], k: int = RETRIEVAL["top_k"]):
        self.store = store
        self.embeddings = embeddings
        self.strategy = strategy
        self.k = k
        self.retriever = RunnableLambda(self.retrieve)
        self.answer_chain = ChatPromptTemplate.from_template(RAG_TEMPLATE) | llm | StrOutputParser()
        self.chain = RunnableParallel(docs=self.retriever, question=RunnablePassthrough()) | RunnablePassthrough.assign(
            answer=RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
            | self.answer_chain
        )

    def retrieve(self, query: str, k: int | None = None) -> list:
        """Retrieve documents for a query."""
        return retrieve(self.store, self.strategy, query, k or self.k)

    def answer_from(self, docs: list, query: str) -> str:
        """Generate an answer from documents that were already retrieved."""
        return self.answer_chain.invoke({"context": format_docs(docs), "question": query})

    def generate(self, query: str) -> dict:
        """Retrieve evidence and generate a cited answer."""
        start, waited = time.perf_counter(), limiter_wait()
        result = self.chain.invoke(query)
        return {
            "answer": result["answer"],
            "evidence": result["docs"],
            "evidence_urls": list(dict.fromkeys(doc.metadata["url"] for doc in result["docs"])),
            "generation_time": time.perf_counter() - start - (limiter_wait() - waited),
        }
