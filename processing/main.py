import os
import time
import hashlib
import functions_framework
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage, documentai
import vertexai
from vertexai.language_models import TextEmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID     = os.environ["GCP_PROJECT_ID"]
GCP_REGION         = os.environ["GCP_REGION"]
DOCAI_PROCESSOR_ID = os.environ["DOCAI_PROCESSOR_ID"]
DOCAI_LOCATION     = os.environ["DOCAI_LOCATION"]
QDRANT_URL         = os.environ["QDRANT_URL"]
QDRANT_API_KEY     = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME    = "rag_documents"

EMBED_BATCH_SIZE  = 5   # pages per Vertex AI call — safe for free + paid quota
QDRANT_BATCH_SIZE = 5   # points per Qdrant upsert — safe for free tier
MAX_RETRIES       = 3   # attempts per batch before giving up
RETRY_DELAYS      = [2, 10, 30]  # seconds between retries

# ── Clients ───────────────────────────────────────────────────────────────────
storage_client = storage.Client(project=GCP_PROJECT_ID)
vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
embed_model    = TextEmbeddingModel.from_pretrained("text-embedding-004")
qdrant_client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr_document(bucket_name: str, file_name: str, mime_type: str) -> list[dict]:
    """
    Run Document AI OCR. Returns list of {page_number, text} dicts.
    Image-only pages that return empty/whitespace text are skipped gracefully.
    """
    docai_client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"}
    )
    processor_name = (
        f"projects/{GCP_PROJECT_ID}/locations/{DOCAI_LOCATION}"
        f"/processors/{DOCAI_PROCESSOR_ID}"
    )

    bucket  = storage_client.bucket(bucket_name)
    blob    = bucket.blob(file_name)
    content = blob.download_as_bytes()

    raw_doc = documentai.RawDocument(content=content, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_doc,
        skip_human_review=True,
    )
    result   = docai_client.process_document(request=request)
    document = result.document

    pages = []
    skipped = 0
    for i, page in enumerate(document.pages):
        page_text = ""
        for segment in page.layout.text_anchor.text_segments:
            start      = int(segment.start_index) if segment.start_index else 0
            end        = int(segment.end_index)
            page_text += document.text[start:end]

        page_text = page_text.strip()

        # Skip image-only or unreadable pages rather than embedding garbage
        if not page_text or len(page_text) < 20:
            print(f"  Page {i+1}: skipped (empty or image-only, got {len(page_text)} chars)")
            skipped += 1
            continue

        pages.append({"page_number": i + 1, "text": page_text})

    print(f"OCR complete: {len(pages)} readable pages, {skipped} skipped.")
    return pages


# ── Embed with retry ──────────────────────────────────────────────────────────

def embed_with_retry(texts: list[str]) -> list[list[float]] | None:
    """
    Embed a batch of texts with exponential backoff retry.
    Returns embeddings list or None if all retries exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            embeddings = embed_model.get_embeddings(texts)
            return [e.values for e in embeddings]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  Embed attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  Embed failed after {MAX_RETRIES} attempts: {e}")
                return None


# ── Upsert with retry ─────────────────────────────────────────────────────────

def upsert_with_retry(points: list[PointStruct]) -> bool:
    """
    Upsert a batch of points to Qdrant with exponential backoff retry.
    Returns True on success, False if all retries exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  Upsert attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  Upsert failed after {MAX_RETRIES} attempts: {e}")
                return False


# ── Deterministic point ID ────────────────────────────────────────────────────

def make_point_id(file_id: str, page_number: int) -> int:
    """
    Deterministic integer ID from file_id + page_number.
    Same file re-processed = same IDs = safe overwrite, no duplicates.
    """
    raw = f"{file_id}_{page_number}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2 ** 63)


# ── Main pipeline ─────────────────────────────────────────────────────────────

@functions_framework.cloud_event
def process_document(cloud_event):
    """
    Entry point — triggered by GCS file upload via Eventarc.
    Each page processed independently: one bad page never kills the rest.
    """
    data        = cloud_event.data
    bucket_name = data["bucket"]
    file_name   = data["name"]

    print(f"[process_document] Starting: {file_name} from bucket: {bucket_name}")

    if file_name.endswith(".pdf"):
        mime_type = "application/pdf"
    elif file_name.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        print(f"Unsupported file type: {file_name}. Skipping.")
        return

    file_url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"
    file_id  = file_name.rsplit(".", 1)[0]

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    print("Step 1: Running Document AI OCR...")
    try:
        pages = ocr_document(bucket_name, file_name, mime_type)
    except Exception as e:
        print(f"OCR failed entirely: {e}. Aborting.")
        raise  # OCR failure is unrecoverable — let Eventarc know

    if not pages:
        print("No readable pages extracted. Skipping.")
        return

    # ── Step 2 + 3: Embed + upsert per batch, page-independently ─────────────
    print(f"Step 2+3: Embedding and storing {len(pages)} pages in batches of {EMBED_BATCH_SIZE}...")

    total_stored  = 0
    total_failed  = 0

    for i in range(0, len(pages), EMBED_BATCH_SIZE):
        batch_pages = pages[i: i + EMBED_BATCH_SIZE]
        batch_texts = [p["text"] for p in batch_pages]
        batch_label = f"pages {batch_pages[0]['page_number']}–{batch_pages[-1]['page_number']}"

        # Embed this batch
        embeddings = embed_with_retry(batch_texts)
        if embeddings is None:
            print(f"  Batch [{batch_label}]: embed failed, skipping these pages.")
            total_failed += len(batch_pages)
            continue  # move on — don't let one bad batch stop the rest

        # Build Qdrant points with deterministic IDs
        points = []
        for page, vector in zip(batch_pages, embeddings):
            points.append(PointStruct(
                id=make_point_id(file_id, page["page_number"]),
                vector=vector,
                payload={
                    "fileId":      file_id,
                    "fileURL":     file_url,
                    "page_number": page["page_number"],
                    "text":        page["text"],
                }
            ))

        # Upsert this batch
        success = upsert_with_retry(points)
        if success:
            total_stored += len(points)
            print(f"  Batch [{batch_label}]: stored {len(points)} pages.")
        else:
            total_failed += len(points)
            print(f"  Batch [{batch_label}]: upsert failed permanently, moving on.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"[process_document] Done: {total_stored} stored, {total_failed} failed — {file_name}")

    if total_failed > 0:
        print(f"WARNING: {total_failed} pages failed permanently. Check logs above.")