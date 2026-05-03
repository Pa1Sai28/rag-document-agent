import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "rag_documents"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection with 768 dimensions (Vertex AI text-embedding-004 output size)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

print(f"Collection '{COLLECTION_NAME}' created successfully.")
print(f"Vector size: 768 dimensions")
print(f"Distance metric: Cosine similarity")