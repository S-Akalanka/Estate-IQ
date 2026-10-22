import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prime_land.chat_service.prompts import REWRITE_TEMPLATE
from prime_land.chat_service.rag_service import RAGService
from prime_land.chat_service.utils import calculate_confidence
from prime_land.config import CONFIG

CRAG = CONFIG["crag"]


class CRAGService:
    """RAG that scores its retrieval and retries with a wider, rewritten search when confidence is low."""

    def __init__(self, rag_service: RAGService, llm):
        self.rag_service = rag_service
        self.rewriter = ChatPromptTemplate.from_template(REWRITE_TEMPLATE) | llm | StrOutputParser()

    def generate(self, query: str, threshold: float = CRAG["confidence_threshold"]) -> dict:
        """Retrieve, correct if confidence is below the threshold, then answer."""
        start = time.perf_counter()
        docs = self.rag_service.retrieve(query)
        confidence_initial = calculate_confidence(docs, query)
        confidence_final = confidence_initial
        corrected = confidence_initial < threshold

        if corrected:
            rewritten = self.rewriter.invoke({"question": query})
            wider = self.rag_service.retrieve(rewritten, CRAG["expanded_k"])
            wider += self.rag_service.retrieve(query, CRAG["expanded_k"])
            unique = {(doc.metadata["url"], doc.page_content): doc for doc in docs + wider}
            ranked = sorted(unique.values(), key=lambda doc: doc.metadata["score"], reverse=True)
            candidate = ranked[: CRAG["expanded_k"]]
            confidence_candidate = calculate_confidence(candidate, query)
            if confidence_candidate > confidence_initial:
                docs, confidence_final = candidate, confidence_candidate

        return {
            "answer": self.rag_service.answer_from(docs, query),
            "evidence": docs,
            "evidence_urls": list(dict.fromkeys(doc.metadata["url"] for doc in docs)),
            "confidence_initial": confidence_initial,
            "confidence_final": confidence_final,
            "correction_applied": corrected,
            "generation_time": time.perf_counter() - start,
        }
