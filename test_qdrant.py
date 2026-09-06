"""
Test Qdrant Cloud connection, collection creation, embedding, upload, and search.

Run: python test_qdrant.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from fastembed import TextEmbedding

COLLECTION = "medverify_test"
MODEL = "BAAI/bge-small-en-v1.5"
TEXT = "Metformin is used to treat type 2 diabetes"


def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("FAIL: QDRANT_URL or QDRANT_API_KEY not found in .env")
        return

    print(f"STEP 1 - Connecting to Qdrant Cloud at {qdrant_url}")
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        info = client.get_collections()
        print(f"SUCCESS: Connected. Existing collections: {len(info.collections)}")
    except Exception as e:
        print(f"FAIL: Could not connect to Qdrant: {e}")
        return

    print(f"STEP 2 - Creating/ensuring collection '{COLLECTION}' (size=384, cosine)")
    try:
        existing = client.collection_exists(COLLECTION)
        if existing:
            client.delete_collection(COLLECTION)
            print("Collection already existed - deleted and will recreate")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"SUCCESS: Collection '{COLLECTION}' created")
    except Exception as e:
        print(f"FAIL: Could not create collection: {e}")
        return

    print(f"STEP 3 - Initializing fastembed TextEmbedding with model '{MODEL}'")
    try:
        embedding_model = TextEmbedding(model_name=MODEL)
        print(f"SUCCESS: Embedding model '{MODEL}' loaded")
    except Exception as e:
        print(f"FAIL: Could not load embedding model: {e}")
        return

    print("STEP 4 - Generating embedding for: 'Metformin is used to treat type 2 diabetes'")
    try:
        vector = next(embedding_model.embed([TEXT])).tolist()
        print(f"SUCCESS: Generated embedding of dimension {len(vector)}")
    except Exception as e:
        print(f"FAIL: Could not generate embedding: {e}")
        return

    print("STEP 5 - Uploading vector with id=1 to collection")
    try:
        client.upsert(
            collection_name=COLLECTION,
            points=[{"id": 1, "vector": vector, "payload": {"text": TEXT}}],
        )
        print("SUCCESS: Vector uploaded")
    except Exception as e:
        print(f"FAIL: Could not upload vector: {e}")
        return

    print("STEP 6 - Searching collection with the same vector")
    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=1,
        )
        top = response.points[0]
        print("SUCCESS: Search returned results")
        print(f"Top result -> id={top.id}, score={top.score:.6f}")
        print(f"Payload text: {top.payload.get('text')}")
    except Exception as e:
        print(f"FAIL: Could not search collection: {e}")
        return

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()