RAG_TEMPLATE = """You are an assistant for Prime Lands, a real estate company in Sri Lanka.

Use only the context below. Cite the source URL in square brackets after each fact.
If the context does not contain the answer, say what is missing.

Context:
{context}

Question: {question}

Answer with 2 to 4 short bullet points and a one sentence summary."""

REWRITE_TEMPLATE = """Rewrite this question about Prime Lands properties in Sri Lanka with more specific search keywords such as property names, locations and property types. Return only the rewritten question.

Question: {question}"""

JUDGE_TEMPLATE = """Rate how well the answer responds to the question on a scale of 1 to 5, where 5 is complete and accurate. Return only the number.

Question: {question}

Answer: {answer}"""
