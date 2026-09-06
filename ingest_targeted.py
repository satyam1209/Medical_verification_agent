"""Targeted openFDA ingestion: metformin + Lactic acidosis reaction.

Run: python ingest_targeted.py
"""

import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient

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

from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance


async def ingest_targeted(topic="metformin", reaction_filter="Lactic acidosis", max_results=10):
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    query_label = f"{topic} (reaction: {reaction_filter})"
    print(f"Fetching openFDA reports for '{query_label}' (max_results={max_results})")

    async with httpx.AsyncClient(timeout=60.0) as client:
        records = await fetch_openfda(client, topic, max_results, reaction_filter=reaction_filter)

    print(f"Found {len(records)} adverse event reports")

    primary_suspect_count = 0
    for record in records:
        metadata = json.loads(record["metadata_json"])
        if metadata.get("is_primary_suspect"):
            primary_suspect_count += 1
    print(f"Metformin is primary suspect (role=1) in {primary_suspect_count} reports")

    init_db()
    session = get_session()
    try:
        qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = await qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in names:
            await qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{QDRANT_COLLECTION}'")

        print(f"Loading embedding model '{EMBED_MODEL}'...")
        embedding_model = TextEmbedding(model_name=EMBED_MODEL)
        print("Embedding model ready.")

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

        print(f"\nSUMMARY: fetched={len(records)} inserted_sqlite={inserted} "
              f"upserted_qdrant={upserted} skipped={skipped} "
              f"primary_suspect={primary_suspect_count}")
    finally:
        session.close()


async def main():
    start = time.time()
    await ingest_targeted()
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())