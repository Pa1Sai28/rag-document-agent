# test_qdrant.py
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv(Path(__file__).resolve().parent / ".env")

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

# Check collection info
info = client.get_collection("rag_documents")
print(f"Total vectors stored: {info.vectors_count}")
print(f"Collection status: {info.status}")

# Show first 5 points
points = client.scroll(
    collection_name="rag_documents",
    limit=5,
    with_payload=True,
    with_vectors=False
)
print("\nSample stored pages:")
for point in points[0]:
    print(f"\n  File: {point.payload['fileId']}")
    print(f"  Page: {point.payload['page_number']}")
    print(f"  Text preview: {point.payload['text'][:100]}...")
