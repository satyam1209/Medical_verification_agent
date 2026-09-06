"""Volume experiment: does ingesting more evidence change the confidence label?

Ingests the vitamin D + calcium fracture topic with max_results=40 per source,
re-runs the synthesis, and compares against the earlier 15-source run.

Run: python volume_experiment.py
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_topic
from synthesize import synthesize_answer

TOPIC = "vitamin D calcium supplementation fracture prevention"
QUESTION = "does vitamin D supplementation combined with calcium prevent fractures in older adults"
PREVIOUS_RESULT = "MODERATE EVIDENCE"
PREVIOUS_SOURCES = 15
NEW_SOURCES = 40


async def ingest():
    print(f"Ingesting topic: {TOPIC!r} (max_results={NEW_SOURCES})")
    await ingest_topic(TOPIC, max_results=NEW_SOURCES)


def main():
    asyncio.run(ingest())

    print("\nGenerating answer...")
    answer = synthesize_answer(QUESTION)
    print(f"\n{answer}\n")

    import re
    matches = re.findall(r"(STRONG|MODERATE|WEAK)\s*EVIDENCE", answer, re.IGNORECASE)
    new_label = f"{matches[-1].upper()} EVIDENCE" if matches else "NONE"

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Previous run ({PREVIOUS_SOURCES} sources ingested): {PREVIOUS_RESULT}")
    print(f"New run ({NEW_SOURCES} sources ingested): {new_label}")
    print("=" * 60)


if __name__ == "__main__":
    main()