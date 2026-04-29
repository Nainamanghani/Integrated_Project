import io
import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from .config import settings

PDF_EXTENSIONS = {".pdf"}


def extract_text_from_pdf_bytes(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        with path.open("rb") as f:
            return extract_text_from_pdf_bytes(f.read())
    raise ValueError(f"Unsupported document type: {suffix}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_document(path: Path) -> Tuple[str, str]:
    text = extract_text_from_file(path)
    return path.name, clean_text(text)


def load_document_bytes(filename: str, data: bytes) -> Tuple[str, str]:
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return filename, clean_text(extract_text_from_pdf_bytes(data))
    raise ValueError(f"Unsupported document type: {suffix}")


def document_to_chunks(source_name: str, text: str) -> List[dict]:
    from .chunker import chunk_text

    chunks = chunk_text(text, max_chars=settings.chunk_size, overlap_chars=settings.chunk_overlap)
    return [
        {
            "id": f"{source_name}-{idx}",
            "source": source_name,
            "content": chunk,
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(chunks)
    ]
