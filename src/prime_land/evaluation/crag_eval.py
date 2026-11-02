import json
import re

import pandas as pd
from groq import RateLimitError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prime_land.chat_service.prompts import JUDGE_TEMPLATE
from prime_land.chat_service.utils import calculate_confidence
from prime_land.config import CONFIG, get_path

EVALUATION = CONFIG["evaluation"]


def judge_score(judge, question: str, answer: str) -> int:
    """Ask the LLM to rate an answer from 1 to 5."""
    reply = judge.invoke({"question": question, "answer": answer})
    match = re.search(r"[1-5]", reply)
    return int(match.group()) if match else 0


def evaluate_crag(rag_service, crag_service, llm) -> tuple[pd.DataFrame, dict]:
    """Compare RAG and CRAG on 20 queries and save the per query table."""
    judge = ChatPromptTemplate.from_template(JUDGE_TEMPLATE) | llm | StrOutputParser()
    questions = [item["question"] for item in EVALUATION["queries"] + EVALUATION["hard_queries"]]

    results_dir = get_path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    progress = results_dir / "crag_progress.jsonl"
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()] if progress.exists() else []
    done = {row["question"] for row in rows}
    if rows:
        print(f"resuming from {len(rows)} saved rows")
    try:
        for number, question in enumerate(questions, 1):
            if question in done:
                continue
            print(f"comparing {number}/{len(questions)}: {question}")
            rag = rag_service.generate(question)
            crag = crag_service.generate(question)
            row = {
                "question": question,
                "rag_confidence": calculate_confidence(rag["evidence"], question),
                "crag_initial_confidence": crag["confidence_initial"],
                "crag_final_confidence": crag["confidence_final"],
                "correction_applied": crag["correction_applied"],
                "rag_quality": judge_score(judge, question, rag["answer"]),
                "crag_quality": judge_score(judge, question, crag["answer"]),
            }
            rows.append(row)
            with open(progress, "a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
        progress.unlink(missing_ok=True)
    except RateLimitError:
        print(f"Groq limit reached, saving {len(rows)} of {len(questions)} rows, rerun to continue")
    if not rows:
        return pd.DataFrame(), {}

    table = pd.DataFrame(rows).round(3)
    table["confidence_gain"] = (table["crag_final_confidence"] - table["rag_confidence"]).round(3)
    table["quality_gain"] = table["crag_quality"] - table["rag_quality"]

    corrected = table[table["correction_applied"]]
    summary = {
        "queries": len(table),
        "correction_frequency": float(table["correction_applied"].mean()),
        "avg_confidence_gain_corrected": float(corrected["confidence_gain"].mean()) if len(corrected) else 0.0,
        "avg_quality_rag": float(table["rag_quality"].mean()),
        "avg_quality_crag": float(table["crag_quality"].mean()),
        "avg_quality_gain": float(table["quality_gain"].mean()),
    }

    table.to_csv(results_dir / "crag_impact.csv", index=False)
    return table, summary
