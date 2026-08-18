import re
import hashlib
from src.config.settings import settings
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitter :

    def split_documents_by_headings(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split Markdown documents by headings while propagating heading context
        across pages belonging to the same file.

        Each output Document:
        - Preserves original metadata.
        - Adds h1, h2, h3, h4 metadata.
        - Inherits headings from previous pages when needed.
        - Adds a hierarchical "section" path.
        """

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )

        result: list[Document] = []

        # Keep heading state separately for each file
        heading_states: dict[str, dict[str, str | None]] = {}

        for document in documents:
            file_id = document.metadata.get("file_id")

            if file_id not in heading_states:
                heading_states[file_id] = {
                    "h1": None,
                    "h2": None,
                    "h3": None,
                    "h4": None,
                }

            current_headers = heading_states[file_id]

            sections = header_splitter.split_text(document.page_content)

            for section in sections:
                section_headers = section.metadata

                # New H1 -> reset all lower levels
                if section_headers.get("h1"):
                    current_headers["h1"] = section_headers["h1"]
                    current_headers["h2"] = None
                    current_headers["h3"] = None
                    current_headers["h4"] = None

                # New H2 -> reset H3 and H4
                if section_headers.get("h2"):
                    current_headers["h2"] = section_headers["h2"]
                    current_headers["h3"] = None
                    current_headers["h4"] = None

                # New H3 -> reset H4
                if section_headers.get("h3"):
                    current_headers["h3"] = section_headers["h3"]
                    current_headers["h4"] = None

                # New H4
                if section_headers.get("h4"):
                    current_headers["h4"] = section_headers["h4"]

                metadata = {
                    **document.metadata,
                    **{
                        level: value
                        for level, value in current_headers.items()
                        if value is not None
                    },
                }

                section_path = " > ".join(
                    metadata[level]
                    for level in ["h1", "h2", "h3", "h4"]
                    if metadata.get(level)
                )

                metadata["section"] = section_path or "unknown"

                result.append(
                    Document(
                        page_content=section.page_content,
                        metadata=metadata,
                    )
                )

        return result

    def split_text_and_tables(self, documents: list[Document]) -> list[Document]:
        """
        Input:
            A list of LangChain Document objects. Each Document contains
            Markdown-formatted content in page_content and its existing metadata.

        Output:
            A list of LangChain Document objects where normal text and Markdown
            tables are separated into independent Documents.

            Each output Document:
                - Preserves all original metadata.
                - Keeps the actual content inside page_content.
                - Adds "content_type" to metadata with either:
                    "text"  -> for normal text content
                    "table" -> for Markdown table content
        """

        result = []

        for document in documents:
            lines = document.page_content.splitlines()

            text_buffer = []
            i = 0

            def flush_text():
                if text_buffer:
                    content = "\n".join(text_buffer).strip()

                    if content:
                        result.append(
                            Document(
                                page_content=content,
                                metadata={
                                    **document.metadata,
                                    "content_type": "text",
                                },
                            )
                        )

                    text_buffer.clear()

            while i < len(lines):

                if i + 1 < len(lines) and "|" in lines[i]:
                    cells = lines[i + 1].strip().strip("|").split("|")

                    is_table = (
                        len(cells) >= 2
                        and all(
                            re.fullmatch(
                                r":?-{3,}:?",
                                cell.strip(),
                            )
                            for cell in cells
                        )
                    )

                    if is_table:
                        flush_text()

                        table_lines = [
                            lines[i],
                            lines[i + 1],
                        ]

                        i += 2

                        while i < len(lines):
                            line = lines[i]

                            if not line.strip() or "|" not in line:
                                break

                            table_lines.append(line)
                            i += 1

                        result.append(
                            Document(
                                page_content="\n".join(table_lines),
                                metadata={
                                    **document.metadata,
                                    "content_type": "table",
                                },
                            )
                        )

                        continue

                text_buffer.append(lines[i])
                i += 1

            flush_text()

        return result

    def split_documents_to_chunks(
        self,
        documents: list[Document],
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        ) -> list[Document]:
        """
        Input:
            A list of LangChain Document objects.
            Each Document must contain "content_type" in its metadata
            with either "text" or "table".

        Output:
            A list of LangChain Document objects where:
                - Documents with content_type="text" are split into smaller chunks.
                - Documents with content_type="table" are preserved as-is.
                - All original metadata is preserved in the output Documents.
        """

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                " ",
                "",
            ],
        )

        result = []

        for document in documents:
            content_type = document.metadata.get("content_type")

            if content_type == "text":
                chunks = splitter.split_documents([document])
                result.extend(chunks)

            elif content_type == "table":
                result.append(document)

            else:
                result.append(document)

        return result

    def add_chunk_indices_ids(self, documents: list[Document]) -> list[Document]:
        """Add positional and unique identifier metadata to all final chunks within
        each original document.

        Adds:
            - chunk_id
            - chunk_index
            - total_chunks
            - previous_chunk_index
            - next_chunk_index
        """

        grouped_documents = {}

        for document in documents:
            document_id = document.metadata.get("file_id")

            grouped_documents.setdefault(document_id, []).append(document)

        result = []

        for document_chunks in grouped_documents.values():
            total_chunks = len(document_chunks)

            for chunk_index, chunk in enumerate(document_chunks):
                # Generate a unique hash for the chunk content
                chunk.metadata["chunk_id"] = hashlib.sha256(
                    chunk.page_content.encode("utf-8")
                ).hexdigest()

                chunk.metadata["chunk_index"] = chunk_index
                chunk.metadata["total_chunks"] = total_chunks

                chunk.metadata["previous_chunk_index"] = (
                    chunk_index - 1 if chunk_index > 0 else None
                )

                chunk.metadata["next_chunk_index"] = (
                    chunk_index + 1 if chunk_index < total_chunks - 1 else None
                )

                result.append(chunk)

        return result
