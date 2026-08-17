import fitz
import uuid
import hashlib
import pymupdf4llm
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from src.indexing.loaders.base import BaseLoader

class PDFLoader(BaseLoader):
    """Load PDF pages with PDF and filesystem metadata."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> list[Document]:
        path = Path(self.path)

        # Extract pages as Markdown.
        pages = pymupdf4llm.to_markdown(
            self.path,
            page_chunks=True,
            header=False,
            footer=False,
            show_progress=False,
        )

        # Open PDF to access document-level metadata.
        pdf = fitz.open(self.path)

        pdf_metadata = pdf.metadata or {}

        # File system metadata.
        stat = path.stat()

        file_metadata = {
            "file_id": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": str(path),
            "source_type": "pdf",
            "file_name": path.name,
            "file_stem": path.stem,
            "file_extension": path.suffix,
            "file_size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "total_pages": pdf.page_count,
        }

        # PDF internal metadata.
        document_metadata = {
            "title": pdf_metadata.get("title"),
            "author": pdf_metadata.get("author"),
            "subject": pdf_metadata.get("subject"),
            "keywords": pdf_metadata.get("keywords"),
            "creator": pdf_metadata.get("creator"),
            "producer": pdf_metadata.get("producer"),
            "creation_date": pdf_metadata.get("creationDate"),
            "modification_date": pdf_metadata.get("modDate"),
            "format": pdf_metadata.get("format"),
            "encryption": pdf_metadata.get("encryption"),
        }

        # Remove empty values.
        document_metadata = {
            key: value
            for key, value in document_metadata.items()
            if value
        }

        documents = []

        for page_number, page in enumerate(pages, start=1):

            # Preserve every metadata field returned by pymupdf4llm.
            page_metadata = page.get(
                "metadata",
                {}
            ).copy()

            metadata = {
                **file_metadata,
                **document_metadata,
                **page_metadata,
                "page": page_number,
            }

            documents.append(
                Document(
                    page_content=page.get("text", ""),
                    metadata=metadata,
                )
            )

        pdf.close()

        return documents
    