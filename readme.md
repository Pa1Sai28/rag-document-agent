# RAG Document Search API

A production-grade Retrieval-Augmented Generation (RAG) system built on Google Cloud Platform. Upload PDF or JPEG documents, automatically extract and embed their content, store embeddings in a vector database, and query documents using natural language — with real-time streaming responses.

---

## What's New

- **Deep health checks** — every API probes its dependencies with real calls, returning per-service status and latency
- **Resilient processing** — per-batch retry with exponential backoff, idempotent point IDs, graceful handling of image-only pages
- **SSE streaming** — query responses stream token by token in real time, exactly like ChatGPT

---

## Architecture

```
[User / Postman]
      │
      ▼
POST /upload
      │
      ▼
Upload API (Cloud Run)
  └── GET /health → probes GCS
      │
      ▼
Google Cloud Storage (GCS)
      │
      ▼  ← Eventarc trigger (automatic)
Processing Pipeline (Cloud Function Gen2)
      │
      ├── Document AI OCR → extract text per page
      ├── Vertex AI Embeddings → 768-dim vector per page
      │   └── batches of 5, retry with 2s/10s/30s backoff
      └── Qdrant Vector DB → batch upsert, idempotent IDs

[User / Postman]
      │
      ▼
POST /query  { "question": "...", "stream": true | false }
      │
      ▼
Query API (Cloud Run)
  └── GET /health → probes Vertex AI + Qdrant + Gemini
      │
      ├── Vertex AI → embed the question
      ├── Qdrant → semantic search (top 3 pages)
      ├── Gemini 2.5 Flash → generate grounded answer
      └── stream=true  → SSE stream  (token / sources / done events)
          stream=false → full JSON   (query + answer + source_pages)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Upload API | Python + FastAPI + Cloud Run |
| File Storage | Google Cloud Storage |
| Processing Pipeline | Cloud Functions Gen2 + Eventarc |
| OCR | Google Document AI |
| Embeddings | Vertex AI (text-embedding-004, 768-dim) |
| Vector Database | Qdrant Cloud |
| LLM | Gemini 2.5 Flash |
| Query API | Python + FastAPI + Cloud Run |

---

## Project Structure

```
RAG_Agent/
├── upload-api/
│   ├── main.py          # FastAPI upload + deep health check
│   ├── Requirements.txt
│   └── Dockerfile
├── processing/
│   ├── main.py          # Resilient Cloud Function pipeline
│   ├── requirements.txt
│   └── setup_qdrant.py
├── query-api/
│   ├── main.py          # FastAPI query (streaming + non-streaming)
│   ├── requirements.txt
│   └── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## Prerequisites

- Google Cloud account with billing enabled
- Python 3.11
- gcloud CLI installed and authenticated
- Qdrant Cloud account (free tier works)
- Google AI Studio account (for Gemini API key)

---

## GCP Services Required

```bash
gcloud services enable \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  documentai.googleapis.com \
  aiplatform.googleapis.com \
  eventarc.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  generativelanguage.googleapis.com
```

---

## Environment Variables

Create a `.env` file at the project root (never commit this file):

```env
GCS_BUCKET_NAME=your-bucket-name
GCP_PROJECT_ID=your-project-id
GCP_PROJECT_NUMBER=your-project-number
GCP_REGION=us-central1
DOCAI_PROCESSOR_ID=your-processor-id
DOCAI_LOCATION=us
VERTEX_AI_LOCATION=us-central1
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GEMINI_API_KEY=your-gemini-api-key
```

---

## Setup Guide

### Step 1 — GCP Setup

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud storage buckets create gs://YOUR_BUCKET_NAME \
  --project=YOUR_PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access

gcloud iam service-accounts create rag-document-sa \
  --display-name "RAG Document Agent Service Account"

for role in roles/storage.admin roles/documentai.apiUser roles/aiplatform.user roles/eventarc.eventReceiver roles/run.invoker roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="$role"
done
```

### Step 2 — Document AI Processor

1. GCP Console → Document AI → Explore Processors
2. Create → Document OCR → Region: `us`
3. Copy the Processor ID to your `.env`

### Step 3 — Qdrant Setup

1. [cloud.qdrant.io](https://cloud.qdrant.io) → create free cluster
2. Copy URL and API key to `.env`
3. Run collection setup:

```bash
cd processing
python setup_qdrant.py
```

### Step 4 — Deploy Upload API

```bash
cd upload-api
gcloud run deploy upload-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=YOUR_BUCKET,GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1
```

### Step 5 — Deploy Processing Pipeline

```bash
cd processing
gcloud functions deploy process-document \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=process_document \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \
  --memory=2Gi \
  --timeout=300s \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,GCS_BUCKET_NAME=YOUR_BUCKET,DOCAI_PROCESSOR_ID=YOUR_PROCESSOR_ID,DOCAI_LOCATION=us,QDRANT_URL=YOUR_QDRANT_URL,QDRANT_API_KEY=YOUR_QDRANT_KEY"
```

### Step 6 — Deploy Query API

```bash
cd query-api
gcloud run deploy query-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,QDRANT_URL=YOUR_QDRANT_URL,QDRANT_API_KEY=YOUR_QDRANT_KEY,GEMINI_API_KEY=YOUR_GEMINI_KEY"
```

---

## API Reference

### Upload API

**GET /health**

Deep health check — probes GCS bucket.

```bash
curl https://YOUR_UPLOAD_API_URL/health
```

```json
{
  "status": "ok",
  "api": "upload-api",
  "services": {
    "gcs": { "status": "ok", "latency_ms": 92 }
  }
}
```

**POST /upload**

```bash
curl -X POST https://YOUR_UPLOAD_API_URL/upload \
  -F "file=@document.pdf"
```

```json
{
  "fileId": "uuid-here",
  "fileURL": "https://storage.googleapis.com/bucket/uuid-here.pdf"
}
```

---

### Query API

**GET /health**

Deep health check — probes Vertex AI, Qdrant, and Gemini.

```bash
curl https://YOUR_QUERY_API_URL/health
```

```json
{
  "status": "ok",
  "api": "query-api",
  "services": {
    "vertex_ai": { "status": "ok", "latency_ms": 345 },
    "qdrant":    { "status": "ok", "latency_ms": 176 },
    "gemini":    { "status": "ok", "latency_ms": 435 }
  }
}
```

Status values: `ok` / `degraded` / `down`. HTTP 503 when any service is `down`.

**POST /query — non-streaming**

```bash
curl -X POST https://YOUR_QUERY_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "stream": false}'
```

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a technique by which...",
  "source_pages": [
    {
      "page_number": 9,
      "fileURL": "https://storage.googleapis.com/bucket/uuid.pdf",
      "relevance_score": 0.6559
    }
  ]
}
```

**POST /query — streaming (SSE)**

```bash
curl -X POST https://YOUR_QUERY_API_URL/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "What is machine learning?", "stream": true}' \
  --no-buffer
```

```
data: {"type": "token",   "content": "Machine learning is"}
data: {"type": "token",   "content": " a technique by which..."}
data: {"type": "sources", "content": [{...}]}
data: {"type": "done"}
```

Three event types: `token` — append to answer, `sources` — render citations, `done` — close connection.

---

## Processing Pipeline — Resilience Details

| Feature | Detail |
|---|---|
| Batch size | 5 pages per Vertex AI call, 5 points per Qdrant upsert |
| Retry policy | 3 attempts per batch — 2s → 10s → 30s backoff |
| Failure isolation | One batch fails → others continue, partial results saved |
| Idempotency | Point ID = hash(fileId + page_number) — re-runs never duplicate |
| Image-only pages | Detected by char count < 20, skipped gracefully with log |

---

## End-to-End Test

```bash
# 1. Upload a PDF
curl -X POST https://YOUR_UPLOAD_API_URL/upload \
  -F "file=@document.pdf"

# 2. Wait 30-60 seconds for processing

# 3. Query with streaming
curl -X POST https://YOUR_QUERY_API_URL/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Your question here", "stream": true}' \
  --no-buffer

# 4. Check processing logs
gcloud functions logs read process-document \
  --region=us-central1 \
  --limit=20
```

---

## Notes

- Document AI free tier: 15 pages per request. Trim large PDFs to 10 pages before uploading.
- Qdrant free tier: 1GB storage — sufficient for hundreds of documents.
- All Cloud Run services scale to zero when idle — no idle costs.
- The `X-Accel-Buffering: no` header on the Query API disables nginx buffering on Cloud Run, which is required for SSE to work correctly.