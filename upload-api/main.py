import uuid
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import storage
from dotenv import load_dotenv

# Always finds .env at project root regardless of where you run from
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI()

BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
storage_client = storage.Client(project=os.environ["GCP_PROJECT_ID"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ("application/pdf", "image/jpeg"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and JPEG files are accepted."
        )

    # Determine extension
    ext = "pdf" if file.content_type == "application/pdf" else "jpg"

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    blob_name = f"{file_id}.{ext}"

    # Upload to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file.file, content_type=file.content_type)

    file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"

    return {
        "fileId": file_id,
        "fileURL": file_url
    }