from __future__ import annotations

import builtins

import pytest

from app.document_parser import (
    ParserDependencyError,
    UnsupportedDocumentTypeError,
    extract_text_from_document,
)


def test_extract_text_from_document_accepts_plain_text() -> None:
    text = extract_text_from_document(
        filename="notes.txt",
        content_type="text/plain",
        raw=b"hello world",
    )
    assert text == "hello world"


def test_extract_text_from_document_rejects_unknown_binary() -> None:
    with pytest.raises(UnsupportedDocumentTypeError):
        extract_text_from_document(
            filename="archive.bin",
            content_type="application/octet-stream",
            raw=b"\x00\x01\x02",
        )


def test_extract_text_from_document_pdf_missing_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pypdf":
            raise ModuleNotFoundError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(ParserDependencyError):
        extract_text_from_document(
            filename="sample.pdf",
            content_type="application/pdf",
            raw=b"%PDF-1.7",
        )
