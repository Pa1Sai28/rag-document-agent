import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Config
GCP_PROJECT_ID  = os.environ["GCP_PROJECT_ID"]
GCP_REGION      = os.environ["GCP_REGION"]
QDRANT_URL      = os.environ["QDRANT_URL"]
QDRANT_API_KEY  = os.environ["QDRANT_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
COLLECTION_NAME = "rag_documents"

# Clients
vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1 — Embed the question using Vertex AI
    question_vector = embed_model.get_embeddings([question])[0].values

    # Step 2 — Semantic search in Qdrant (top 3)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        limit=3,
        with_payload=True
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # Step 3 — Build prompt with retrieved pages as context
    context = ""
    for result in results:
        context += f"\n--- Page {result.payload['page_number']} from {result.payload['fileURL']} ---\n"
        context += result.payload["text"]
        context += "\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find relevant information in the documents."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4 — LLM call via Gemini API
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
        )
    answer = response.text.strip()

    # Step 5 — Build source pages list
    source_pages = [
        {
            "page_number":     result.payload["page_number"],
            "fileURL":         result.payload["fileURL"],
            "relevance_score": round(result.score, 4)
        }
        for result in results
    ]

    return {
        "query":        question,
        "answer":       answer,
        "source_pages": source_pages
    }