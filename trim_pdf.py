# trim_pdfs.py
import os
from pypdf import PdfReader, PdfWriter

input_folder = "Documents_rag"
output_folder = "Documents_rag_trimmed"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        reader = PdfReader(f"{input_folder}/{filename}")
        writer = PdfWriter()
        pages_to_take = min(10, len(reader.pages))
        for i in range(pages_to_take):
            writer.add_page(reader.pages[i])
        with open(f"{output_folder}/{filename}", "wb") as f:
            writer.write(f)
        print(f"Trimmed {filename}: {len(reader.pages)} → {pages_to_take} pages")
