# RAG Document Search API

A fully functional Retrieval-Augmented Generation (RAG) system built on Google Cloud Platform. Upload PDF or JPEG documents, automatically extract and embed their content, store embeddings in a vector database, and query documents using natural language.

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
      │
      ▼
Google Cloud Storage (GCS)
      │
      ▼  ← Eventarc trigger (automatic)
Processing Pipeline (Cloud Function)
      │
      ├── Document AI OCR → extract text per page
      ├── Vertex AI Embeddings → 768-dim vector per page
      └── Qdrant Vector DB → store vectors + metadata
      
[User / Postman]
      │
      ▼
POST /query
      │
      ▼
Query API (Cloud Run)
      │
      ├── Vertex AI → embed the question
      ├── Qdrant → semantic search (top 3 pages)
      ├── Gemini 2.5 Flash → generate grounded answer
      └── Return: query + answer + source pages
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
| LLM | Gemini 2.5 Flash (Google AI Studio) |
| Query API | Python + FastAPI + Cloud Run |

---

## Project Structure

```
RAG_Agent/
├── upload-api/
│   ├── main.py          # FastAPI upload endpoint
│   ├── requirements.txt
│   └── Dockerfile
├── processing/
│   ├── main.py          # Cloud Function pipeline
│   ├── requirements.txt
│   └── setup_qdrant.py  # One-time collection setup
├── query-api/
│   ├── main.py          # FastAPI query endpoint
│   ├── requirements.txt
│   └── Dockerfile
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

---

## Prerequisites

- Google Cloud account with billing enabled
- Python 3.11
- gcloud CLI installed and authenticated
- Qdrant Cloud account (free tier)
- Google AI Studio account (for Gemini API key)

---

## GCP Services Required

Enable these APIs in your GCP project:

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
UPLOAD_API_URL=https://your-upload-api.run.app
```

---

## Setup Guide

### Step 1 — GCP Setup

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create GCS bucket
gcloud storage buckets create gs://YOUR_BUCKET_NAME \
  --project=YOUR_PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access

# Create service account
gcloud iam service-accounts create rag-document-sa \
  --display-name "RAG Document Agent Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/documentai.apiUser"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/eventarc.eventReceiver"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"
```

### Step 2 — Document AI Processor

1. Go to GCP Console → Document AI → Explore Processors
2. Create Processor → Document OCR
3. Name: `rag-ocr-processor`, Region: `us`
4. Copy the Processor ID to your `.env`

### Step 3 — Qdrant Setup

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create free cluster → copy URL and API key to `.env`
3. Run the collection setup script:

```bash
cd processing
python setup_qdrant.py
```

### Step 4 — Virtual Environment

```bash
python3.11 -m venv rag_agent_env
source rag_agent_env/bin/activate
```

### Step 5 — Deploy Upload API

```bash
cd upload-api
pip install -r requirements.txt

# Test locally
uvicorn main:app --reload --port 8080

# Deploy to Cloud Run
gcloud run deploy upload-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --service-account=rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCS_BUCKET_NAME=YOUR_BUCKET,GCP_PROJECT_ID=YOUR_PROJECT_ID \
  --min-instances=0 \
  --max-instances=1
```

### Step 6 — Deploy Processing Pipeline

```bash
cd processing
pip install -r requirements.txt

# Grant Eventarc permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:service-YOUR_PROJECT_NUMBER@gs-project-accounts.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:service-YOUR_PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator"

# Deploy
gcloud functions deploy process-document \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=process_document \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \
  --service-account=rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,DOCAI_PROCESSOR_ID=YOUR_PROCESSOR_ID,DOCAI_LOCATION=us,QDRANT_URL=YOUR_QDRANT_URL,QDRANT_API_KEY=YOUR_QDRANT_KEY \
  --memory=2Gi \
  --timeout=300s \
  --min-instances=0 \
  --max-instances=1
```

### Step 7 — Deploy Query API

```bash
cd query-api
pip install -r requirements.txt

# Test locally
uvicorn main:app --reload --port 8082

# Deploy to Cloud Run
gcloud run deploy query-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --service-account=rag-document-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,QDRANT_URL=YOUR_QDRANT_URL,QDRANT_API_KEY=YOUR_QDRANT_KEY,GEMINI_API_KEY=YOUR_GEMINI_KEY \
  --min-instances=0 \
  --max-instances=1
```

---

## API Reference

### Upload API

**POST /upload**

Upload a PDF or JPEG file.

```bash
curl -X POST https://YOUR_UPLOAD_API_URL/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "fileId": "uuid-here",
  "fileURL": "https://storage.googleapis.com/bucket/uuid-here.pdf"
}
```

**GET /health**

```bash
curl https://YOUR_UPLOAD_API_URL/health
```

Response: `{"status": "ok"}`

---

### Query API

**POST /query**

Query documents using natural language.

```bash
curl -X POST https://YOUR_QUERY_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

Response:
```json
{
  "query": "What is artificial intelligence?",
  "answer": "Artificial intelligence (AI) is...",
  "source_pages": [
    {
      "page_number": 1,
      "fileURL": "https://storage.googleapis.com/bucket/uuid.pdf",
      "relevance_score": 0.7546
    }
  ]
}
```

**GET /health**

```bash
curl https://YOUR_QUERY_API_URL/health
```

Response: `{"status": "ok"}`

---

## Testing End-to-End

```bash
# 1. Upload a PDF
curl -X POST https://YOUR_UPLOAD_API_URL/upload \
  -F "file=@document.pdf"

# 2. Wait 30-60 seconds for processing

# 3. Query the document
curl -X POST https://YOUR_QUERY_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here?"}'
```

---

## Notes

- Document AI free tier processes up to 30 pages per request. Trim large PDFs before uploading.
- Qdrant free tier provides 1GB storage — sufficient for hundreds of documents.
- Gemini 2.5 Flash is used for LLM generation — cost is negligible for demo usage.
- All Cloud Run services scale to zero when not in use — no idle costs.