import os
import uuid
import functions_framework
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage, documentai
import vertexai
from vertexai.language_models import TextEmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Config
GCP_PROJECT_ID     = os.environ["GCP_PROJECT_ID"]
GCP_REGION         = os.environ["GCP_REGION"]
DOCAI_PROCESSOR_ID = os.environ["DOCAI_PROCESSOR_ID"]
DOCAI_LOCATION     = os.environ["DOCAI_LOCATION"]
QDRANT_URL         = os.environ["QDRANT_URL"]
QDRANT_API_KEY     = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME    = "rag_documents"

# Clients
storage_client = storage.Client(project=GCP_PROJECT_ID)
qdrant_client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")


def ocr_document(bucket_name: str, file_name: str, mime_type: str) -> list[str]:
    """Send file to Document AI and return list of page texts."""
    docai_client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"}
    )
    processor_name = (
        f"projects/{GCP_PROJECT_ID}/locations/{DOCAI_LOCATION}"
        f"/processors/{DOCAI_PROCESSOR_ID}"
    )

    # Download file from GCS
    bucket  = storage_client.bucket(bucket_name)
    blob    = bucket.blob(file_name)
    content = blob.download_as_bytes()

    # Simple clean request — no field_mask, no extra options
    raw_doc = documentai.RawDocument(content=content, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_doc,
        skip_human_review=True,
    )
    result   = docai_client.process_document(request=request)
    document = result.document

    # Extract text per page
    pages_text = []
    for page in document.pages:
        page_text = ""
        for segment in page.layout.text_anchor.text_segments:
            start      = int(segment.start_index) if segment.start_index else 0
            end        = int(segment.end_index)
            page_text += document.text[start:end]
        if page_text.strip():
            pages_text.append(page_text.strip())

    return pages_text


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Vertex AI text-embedding-004."""
    embeddings = embed_model.get_embeddings(texts)
    return [e.values for e in embeddings]


def store_in_qdrant(pages: list[str], embeddings: list[list[float]],
                    file_id: str, file_url: str):
    """Store page embeddings and metadata in Qdrant."""
    points = []
    for i, (text, vector) in enumerate(zip(pages, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "fileId":      file_id,
                "fileURL":     file_url,
                "page_number": i + 1,
                "text":        text
            }
        ))
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} page embeddings in Qdrant.")


@functions_framework.cloud_event
def process_document(cloud_event):
    """Entry point — triggered automatically by GCS file upload via Eventarc."""
    data        = cloud_event.data
    bucket_name = data["bucket"]
    file_name   = data["name"]

    print(f"Processing file: {file_name} from bucket: {bucket_name}")

    # Determine MIME type
    if file_name.endswith(".pdf"):
        mime_type = "application/pdf"
    elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        mime_type = "image/jpeg"
    else:
        print(f"Unsupported file type: {file_name}. Skipping.")
        return

    file_url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"
    file_id  = file_name.rsplit(".", 1)[0]

    try:
        # Step 1 — OCR
        print("Step 1: Running Document AI OCR...")
        pages = ocr_document(bucket_name, file_name, mime_type)
        print(f"Extracted {len(pages)} pages.")

        if not pages:
            print("No text extracted. Skipping.")
            return

        # Step 2 — Embed in batches of 5
        print("Step 2: Embedding pages with Vertex AI...")
        all_embeddings = []
        batch_size = 5
        for i in range(0, len(pages), batch_size):
            batch            = pages[i:i + batch_size]
            batch_embeddings = embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"  Embedded pages {i+1} to {min(i+batch_size, len(pages))}")

        # Step 3 — Store in Qdrant
        print("Step 3: Storing embeddings in Qdrant...")
        store_in_qdrant(pages, all_embeddings, file_id, file_url)

        print(f"Successfully processed {file_name} — {len(pages)} pages embedded and stored.")

    except Exception as e:
        print(f"ERROR processing {file_name}: {str(e)}")
        raise