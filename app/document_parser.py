from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class UnsupportedDocumentTypeError(ValueError):
    pass


TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".log",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sql",
    ".sh",
    ".toml",
    ".ini",
    ".cfg",
}


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        parts.append((page.extract_text() or "").strip())
    return "\n\n".join(part for part in parts if part)


def extract_text_from_document(filename: str | None, content_type: str | None, raw: bytes) -> str:
    if not raw:
        raise ValueError("Uploaded file is empty")

    ext = Path(filename or "").suffix.lower()
    mime = (content_type or "").lower()

    if ext == ".pdf" or "pdf" in mime:
        text = _extract_pdf_text(raw)
    elif ext in TEXT_FILE_EXTENSIONS or mime.startswith("text/") or mime in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }:
        text = _decode_text(raw)
    else:
        supported = "PDF and common text files (.txt, .md, .csv, .json, .xml, code files, etc.)"
        raise UnsupportedDocumentTypeError(f"Unsupported file type. Supported: {supported}")

    text = text.strip()
    if not text:
        raise ValueError("Could not extract readable text from the uploaded file")
    return text
