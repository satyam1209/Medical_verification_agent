"""Retrieve relevant evidence from the Qdrant knowledge base.

Run: python retrieve.py
"""

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from fastembed import TextEmbedding

from check_retraction import check_retraction_status
from freshness import freshness_assessment as check_freshness
from evidence import COVERAGE_TARGETED_SUFFIXES, assess_coverage, characterize_record

QDRANT_COLLECTION = "medverify_kb"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DISPLAY_TEXT_CHARS = 300
SEARCH_CACHE_TTL_SECONDS = 300

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embedding_model = TextEmbedding(model_name=EMBED_MODEL)

_search_cache = {}


def clear_search_cache():
    """Discard cached search results (call after a forced re-ingestion)."""
    _search_cache.clear()


def _embed(text):
    return next(embedding_model.embed([text])).tolist()


def search_kb(query: str, top_k: int = 5, source_filter: str = None, truncate: bool = True):
    cache_key = (query, top_k, source_filter, truncate)
    hit = _search_cache.get(cache_key)
    if hit and (time.time() - hit[0]) < SEARCH_CACHE_TTL_SECONDS:
        return hit[1]

    results = _search_kb_uncached(query, top_k, source_filter, truncate)
    results = _exclude_retracted_pubmed(results)
    _search_cache[cache_key] = (time.time(), results)
    return results


def _search_kb_uncached(query: str, top_k: int = 5, source_filter: str = None,
                        truncate: bool = True, retrieval_stage: str = "semantic"):
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
        record["retrieval_stage"] = retrieval_stage
        results.append(record)

    return results


def search_kb_staged(query: str, top_k: int = 8, truncate: bool = False,
                     extra_per_type: int = 5):
    """Multi-stage retrieval (evidence rule #3).

    Stage 1: semantic search for the query.
    Stage 2: coverage check on the stage-1 set; for important evidence types
    that are MISSING (SR/MA, RCT, observational, guidelines, safety), run
    targeted semantic queries and merge new (source, id) records, up to a
    bounded budget.
    A single retraction-exclusion pass runs over the merged set.
    """
    cache_key = ("staged", query, top_k, truncate, extra_per_type)
    hit = _search_cache.get(cache_key)
    if hit and (time.time() - hit[0]) < SEARCH_CACHE_TTL_SECONDS:
        return hit[1]

    stage_one = _search_kb_uncached(query, top_k, None, truncate)
    coverage = assess_coverage(stage_one)

    seen = {(r["source"], r["id"]) for r in stage_one}
    extra = []
    for type_key in ("SR/MA", "RCT", "OBSERVATIONAL", "GUIDELINES", "SAFETY"):
        if coverage["present"].get(type_key):
            continue
        suffix = COVERAGE_TARGETED_SUFFIXES.get(type_key)
        if not suffix:
            continue
        targeted_query = f"{query} {suffix}"
        candidates = _search_kb_uncached(
            targeted_query, extra_per_type, None, truncate,
            retrieval_stage=f"targeted:{type_key}",
        )
        for c in candidates:
            key = (c["source"], c["id"])
            if key in seen:
                continue
            seen.add(key)
            extra.append(c)
            if len(extra) >= 12:
                break

    merged = stage_one + extra
    merged = _exclude_retracted_pubmed(merged)
    _search_cache[cache_key] = (time.time(), merged)
    return merged


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