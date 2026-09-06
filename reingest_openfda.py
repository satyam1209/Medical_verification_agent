"""Delete existing openfda records from SQLite + Qdrant and re-ingest them for a topic.

Run: python reingest_openfda.py
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)
from fastembed import TextEmbedding

from db_setup import init_db, get_session, now_str
from schema import MedicalRecord
from ingest import (
    QDRANT_COLLECTION,
    EMBED_MODEL,
    fetch_openfda,
    _embed_text,
    _point_id,
)


async def reingest_openfda(topic="metformin", max_results=15):
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print(f"STEP 1 - Deleting existing openfda records from SQLite")
    init_db()
    session = get_session()
    deleted_sql = session.query(MedicalRecord).filter(MedicalRecord.source == "openfda").delete()
    session.commit()
    print(f"Deleted {deleted_sql} openfda records from SQLite")

    print("STEP 2 - Deleting existing openfda points from Qdrant")
    qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    openfda_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value="openfda"))])
    del_result = await qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=FilterSelector(filter=openfda_filter),
    )
    print(f"Qdrant deletion status: {del_result.status}")

    init_db()
    print(f"STEP 3 - Re-fetching openfda records for '{topic}' (max_results={max_results})")
    async with httpx.AsyncClient(timeout=60.0) as client:
        records = await fetch_openfda(client, topic, max_results)
    print(f"Fetched {len(records)} openfda records")

    print(f"STEP 4 - Re-inserting into SQLite and upserting into Qdrant")
    embedding_model = TextEmbedding(model_name=EMBED_MODEL)

    inserted = 0
    upserted = 0
    skipped = 0

    for record in records:
        try:
            if not record.get("text") or not str(record["text"]).strip():
                print(f"SKIP (empty text): {record['id']}")
                skipped += 1
                continue

            existing = session.query(MedicalRecord).filter_by(
                id=record["id"], source=record["source"]
            ).first()
            if existing:
                print(f"SKIP (duplicate): {record['id']}")
                skipped += 1
                continue

            session.add(MedicalRecord(
                id=record["id"],
                source=record["source"],
                type=record["type"],
                title=record["title"],
                text=record["text"],
                publish_date=record["publish_date"],
                last_updated=record["last_updated"],
                url=record["url"],
                metadata_json=record["metadata_json"],
                ingestion_date=now_str(),
            ))
            session.commit()
            inserted += 1

            vector = await asyncio.to_thread(_embed_text, embedding_model, record["text"])

            payload = dict(record)
            payload["topic"] = topic
            point = PointStruct(
                id=_point_id(record["id"], record["source"]),
                vector=vector,
                payload=payload,
            )
            await qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])
            upserted += 1
        except Exception as e:
            print(f"WARNING: failed to ingest record {record['id']}: {e}")
            skipped += 1

    await qdrant.close()
    session.close()

    print(f"\nSUMMARY: fetched={len(records)} inserted_sqlite={inserted} "
          f"upserted_qdrant={upserted} skipped={skipped}")


async def main():
    start = time.time()
    await reingest_openfda("metformin", max_results=15)
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())