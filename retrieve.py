"""Retrieve relevant evidence from the Qdrant knowledge base.

Run: python retrieve.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from fastembed import TextEmbedding

from check_retraction import check_retraction_status
from freshness import freshness_assessment as check_freshness
from evidence import characterize_record

QDRANT_COLLECTION = "medverify_kb"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DISPLAY_TEXT_CHARS = 300

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embedding_model = TextEmbedding(model_name=EMBED_MODEL)


def _embed(text):
    return next(embedding_model.embed([text])).tolist()


def search_kb(query: str, top_k: int = 5, source_filter: str = None, truncate: bool = True):
    query_vector = _embed(query)

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )
            ]
        )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    results = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text", "") or ""
        if truncate and len(text) > DISPLAY_TEXT_CHARS:
            text = text[:DISPLAY_TEXT_CHARS] + "..."
        record = {
            "id": payload.get("id"),
            "score": round(point.score, 6),
            "source": payload.get("source"),
            "type": payload.get("type"),
            "title": payload.get("title"),
            "text": text,
            "url": payload.get("url"),
            "publish_date": payload.get("publish_date"),
            "metadata_json": payload.get("metadata_json"),
            "freshness_label": check_freshness(payload.get("publish_date"))["category"],
        }
        record.update(characterize_record(record))
        results.append(record)

    results = _exclude_retracted_pubmed(results)
    return results


def _exclude_retracted_pubmed(results):
    pubmed = [r for r in results if r.get("source") == "pubmed"]
    if not pubmed:
        return results

    async def _check_all(records):
        return await asyncio.gather(
            *(check_retraction_status(rec["id"]) for rec in records)
        )

    verdicts = asyncio.run(_check_all(pubmed))
    excluded_ids = {
        id(rec) for rec, v in zip(pubmed, verdicts) if v.get("is_retracted")
    }
    for rec, v in zip(pubmed, verdicts):
        if v.get("is_retracted"):
            print(f"WARNING: Excluded retracted paper PMID {rec['id']} from results")

    return [
        r for r in results
        if r.get("source") != "pubmed" or id(r) not in excluded_ids
    ]


def print_results(results, query):
    print(f"\nTop {len(results)} results for: {query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r['score']}] [{r['source']}] {r['title']}")
        print(f"   {r['text']}")
        print(f"   URL: {r['url']}\n")


def run_test():
    print("TEST: search_kb('metformin side effects', top_k=5, source_filter='openfda')")
    results = search_kb("metformin side effects", top_k=5, source_filter="openfda")
    print_results(results, "metformin side effects")


def run_test2():
    print("TEST: search_kb('metformin lactic acidosis', top_k=5, source_filter='openfda')")
    results = search_kb("metformin lactic acidosis", top_k=5, source_filter="openfda", truncate=False)
    print_results(results, "metformin lactic acidosis")


if __name__ == "__main__":
    run_test()
    run_test2()

    if "--test" in __import__("sys").argv:
        print("Test run complete.")
        __import__("sys").exit(0)

    print("Evidence retrieval from medverify_kb. Type 'exit' to quit.")
    print(f"Model: {EMBED_MODEL}\n")
    while True:
        query = input("Enter query: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        try:
            results = search_kb(query, top_k=5)
            print_results(results, query)
        except Exception as e:
            print(f"Error during search: {e}")