"""Verify ingestion by counting records in SQLite and points in Qdrant.

Run: python verify_ingestion.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func

from qdrant_client import AsyncQdrantClient

from db_setup import get_session
from schema import MedicalRecord

QDRANT_COLLECTION = "medverify_kb"


def check_sqlite():
    print("\n--- SQLite (medverify.db) ---")
    session = get_session()
    try:
        total = session.query(func.count(MedicalRecord.id)).scalar()
        print(f"Total records in medical_records: {total}")

        by_source = (
            session.query(MedicalRecord.source, func.count(MedicalRecord.id))
            .group_by(MedicalRecord.source)
            .order_by(MedicalRecord.source)
            .all()
        )
        for source, count in by_source:
            print(f"  {source}: {count}")
    finally:
        session.close()


async def check_qdrant():
    print("\n--- Qdrant (medverify_kb) ---")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("FAIL: QDRANT_URL or QDRANT_API_KEY not found in .env")
        return

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    try:
        info = await client.get_collection(collection_name=QDRANT_COLLECTION)
        print(f"Total points in {QDRANT_COLLECTION}: {info.points_count}")
    finally:
        await client.close()


if __name__ == "__main__":
    check_sqlite()
    asyncio.run(check_qdrant())