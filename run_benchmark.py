"""Run the medical evidence pipeline against the Cochrane benchmark set.

For each test case in cochrane_benchmark.json:
   1. Ensure relevant data is ingested (ingest_topic, safely skips dups)
   2. Synthesize an answer (synthesize_answer)
   3. Extract the pipeline's confidence label
   4. Compare against the expected Cochrane confidence (MATCH / MISMATCH)

Prints a summary table + score, then saves full results to benchmark_results.json.

Run: python run_benchmark.py
"""

import asyncio
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_topic
from synthesize import synthesize_answer

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "cochrane_benchmark.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")

# Map the pipeline's full label back to the Cochrane-style single word
LABEL_TO_WORD = {
    "STRONG EVIDENCE": "STRONG",
    "MODERATE EVIDENCE": "MODERATE",
    "WEAK EVIDENCE": "WEAK",
}

WORD_TO_LABEL = {v: k for k, v in LABEL_TO_WORD.items()}


def load_benchmark():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


async def ensure_ingested(topic):
    try:
        await ingest_topic(topic, max_results=15)
    except Exception as exc:
        print(f"[ingest] WARNING: ingest_topic failed ({exc}); "
              f"continuing with whatever is already in the KB")


def extract_confidence(answer_text):
    """Return the FINAL Confidence label from the answer.

    The authoritative label is on the last 'Confidence:' line.  The new answer
    schema may also contain 'Retrieved evidence certainty: STRONG EVIDENCE',
    so a plain whole-text regex would select the wrong value; anchor on the
    final Confidence line and fall back to last-position matching for legacy
    answers.
    """
    if not answer_text:
        return None
    final = re.findall(
        r"Confidence:\s*(OVERALL CERTAINTY CANNOT BE DETERMINED FROM THE "
        r"RETRIEVED EVIDENCE|STRONG EVIDENCE|MODERATE EVIDENCE|WEAK EVIDENCE)",
        answer_text,
        re.IGNORECASE,
    )
    if final:
        label = final[-1].upper()
        return "CANNOT DETERMINE" if label.startswith("OVERALL") else label

    matches = re.findall(r"(STRONG|MODERATE|WEAK)\s*EVIDENCE", answer_text, re.IGNORECASE)
    if not matches:
        return None
    last_word = matches[-1].upper()
    return f"{last_word} EVIDENCE"


def run_case(test_case):
    topic = test_case["topic_for_ingestion"]
    question = test_case["question"]

    print(f"\n=== Test case {test_case['id']}: {question} ===")

    print(f"\n[ingest] Ensuring data for topic: {topic!r}")
    asyncio.run(ensure_ingested(topic))

    print(f"[synthesize] Generating answer for question: {question!r}")
    try:
        answer = synthesize_answer(question)
    except Exception as exc:
        print(f"[synthesize] WARNING: synthesize_answer failed: {exc}")
        answer = None

    pipeline_label = extract_confidence(answer) if answer else None
    pipeline_word = LABEL_TO_WORD.get(pipeline_label) if pipeline_label else None
    cochrane = test_case["cochrane_confidence"]
    cochrane_label = WORD_TO_LABEL.get(cochrane, cochrane)

    match = (pipeline_word == cochrane) if pipeline_word else False
    match_status = "MATCH" if match else "MISMATCH"

    print(f"[parse] Pipeline confidence: {pipeline_label!r} "
          f"| Cochrane confidence: {cochrane_label!r} "
          f"| Result: {match_status}")

    return {
        "id": test_case["id"],
        "question": question,
        "topic_for_ingestion": topic,
        "pipeline_answer": answer,
        "pipeline_confidence": pipeline_label,
        "pipeline_confidence_word": pipeline_word,
        "cochrane_confidence": cochrane_label,
        "match": match,
        "match_status": match_status,
    }


def main():
    benchmark = load_benchmark()
    results = []

    for test_case in benchmark:
        try:
            results.append(run_case(test_case))
        except Exception as exc:
            print(f"\n[error] Test case {test_case['id']} failed: {exc}")
            results.append({
                "id": test_case["id"],
                "question": test_case["question"],
                "topic_for_ingestion": test_case["topic_for_ingestion"],
                "pipeline_answer": None,
                "pipeline_confidence": None,
                "pipeline_confidence_word": None,
                "cochrane_confidence": test_case["cochrane_confidence"],
                "match": False,
                "match_status": "MISMATCH",
                "error": str(exc),
            })

    matched = sum(1 for r in results if r["match"])
    total = len(results)
    pct = round(100.0 * matched / total, 1) if total else 0.0

    print("\n\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'ID':<4} | {'Question':<50} | {'Your Conf':<16} | {'Cochrane':<16} | {'Match'}"
    print(header)
    print("-" * len(header))
    for r in results:
        q = (r["question"] or "")[:50]
        yours = (r["pipeline_confidence"] or "NONE")[:16]
        coch = (r["cochrane_confidence"] or "NONE")[:16]
        print(f"{r['id']:<4} | {q:<50} | {yours:<16} | {coch:<16} | {r['match_status']}")

    print("-" * len(header))
    print(f"\nSCORE: {matched} out of {total} matched ({pct}%)")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()