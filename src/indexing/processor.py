import time
import hashlib
from typing import Callable, Any
from langchain_core.documents import Document
from src.indexing.loaders import (PDFLoader)
from src.indexing.loader import DocumentLoader
from src.indexing.cleaner import DocumentCleaner
from src.indexing.splitter import DocumentSplitter
from src.indexing.vector_store import VectorStore
from src.indexing.embeddings import Embedding

class DocumentProcessor :

    def enrich_documents(self, documents: list[Document]) -> list[Document]:
        result = []

        for document in documents:
            section = document.metadata.get("section", "Unknown")
            # page = document.metadata.get("page", "Unknown")

            context = (
                f"Section: {section}\n"
            )

            result.append(
                Document(
                    page_content=f"{context}\n{document.page_content}",
                    metadata=document.metadata.copy(),
                )
            )

        return result

    def deduplicate_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Input:
            A list of LangChain Document objects.

        Output:
            A list of Document objects with duplicate content removed.
            The first occurrence of each unique document is preserved
            together with all of its original metadata.

        The comparison is performed on normalized page_content by:
            - Converting text to lowercase.
            - Removing differences in extra whitespace.
            - Generating a SHA-256 hash for efficient duplicate detection.
        """

        seen = set()
        result = []

        for document in documents:

            normalized_content = " ".join(
                document.page_content.lower().split()
            )

            content_hash = hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                result.append(document)

        return result

    def document_processing_pipeline(self, path : str, vector_store:VectorStore ) :
        documents = DocumentLoader().load_data(PDFLoader(path=path))
        clean_documents = DocumentCleaner().clean_documents(documents=documents)
        sections = DocumentSplitter().split_documents_by_headings(clean_documents)
        blocks = DocumentSplitter().split_text_and_tables(sections)
        chunks = DocumentSplitter().split_documents_to_chunks(blocks)
        chunks = self.enrich_documents(chunks) 
        final_chunks = self.deduplicate_documents(chunks)
        final_chunks = DocumentSplitter().add_chunk_indices(chunks)

        ids = vector_store.generate_chunks_ids(final_chunks)
        ids = vector_store.add_new_documents(vector_store=vector_store.get_vector_store(), chunks=final_chunks, ids=ids)

        return 'Document Added.'

    def _format_time(self, seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.2f} ms"

        return f"{seconds:.2f} sec"

    def _track_step(
        self,
        step_name: str,
        func: Callable[[], Any],
        pipeline_start: float
    ) -> Any:

        print(f"\n{'=' * 70}")
        print(f"[START] {step_name}")

        step_start = time.perf_counter()

        try:
            result = func()

            step_time = time.perf_counter() - step_start
            total_time = time.perf_counter() - pipeline_start

            print(f"[DONE]  {step_name}")
            print(f"Step Time  : {self._format_time(step_time)}")
            print(f"Total Time : {total_time:.2f} sec")
            print(f"Type       : {type(result).__name__}")

            if hasattr(result, "__len__"):
                try:
                    print(f"Length     : {len(result)}")
                except TypeError:
                    pass

            return result

        except Exception as e:
            step_time = time.perf_counter() - step_start
            total_time = time.perf_counter() - pipeline_start

            print(f"[ERROR] {step_name}")
            print(f"Step Time  : {self._format_time(step_time)}")
            print(f"Total Time : {total_time:.2f} sec")
            print(f"Error      : {type(e).__name__}: {e}")

            raise

    def document_processing_pipeline_with_track(
        self,
        path: str,
        embedding_model
    ):
        pipeline_start = time.perf_counter()

        print("\n" + "=" * 70)
        print("DOCUMENT PROCESSING PIPELINE STARTED")
        print("=" * 70)

        documents = self._track_step(
            "1. Load PDF",
            lambda: DocumentLoader().load_data(
                PDFLoader(path=path)
            ),
            pipeline_start
        )

        clean_documents = self._track_step(
            "2. Clean Documents",
            lambda: DocumentCleaner().clean_documents(
                documents=documents
            ),
            pipeline_start
        )

        sections = self._track_step(
            "3. Split Documents By Headings",
            lambda: DocumentSplitter().split_documents_by_headings(
                clean_documents
            ),
            pipeline_start
        )

        blocks = self._track_step(
            "4. Split Text And Tables",
            lambda: DocumentSplitter().split_text_and_tables(
                sections
            ),
            pipeline_start
        )

        chunks = self._track_step(
            "5. Split Documents Into Chunks",
            lambda: DocumentSplitter().split_documents_to_chunks(
                blocks
            ),
            pipeline_start
        )

        chunks = self._track_step(
            "6. Enrich Chunks",
            lambda: self.enrich_documents(chunks),
            pipeline_start
        )

        chunks = self._track_step(
            "7. Deduplicate Chunks",
            lambda: self.deduplicate_documents(chunks),
            pipeline_start
        )

        final_chunks = self._track_step(
            "8. Add Chunk Indices",
            lambda: DocumentSplitter().add_chunk_indices(chunks),
            pipeline_start
        )

        vector_store = self._track_step(
            "9. Initialize Vector Store",
            lambda: VectorStore(
                embedding_model=embedding_model
            ),
            pipeline_start
        )

        ids = self._track_step(
            "10. Generate Chunk IDs",
            lambda: vector_store.generate_chunks_ids(final_chunks),
            pipeline_start
        )

        added_ids = self._track_step(
            "11. Add Documents To Vector Store",
            lambda: vector_store.add_new_documents(
                vector_store=vector_store.get_vector_store(),
                chunks=final_chunks,
                ids=ids
            ),
            pipeline_start
        )

        total_time = time.perf_counter() - pipeline_start

        print("\n" + "=" * 70)
        print("INDEXING DONE")
        print(f"Total Pipeline Time : {total_time:.2f} sec")
        print(f"Final Chunks        : {len(final_chunks)}")
        print(f"Output Type         : {type(added_ids).__name__}")
        print("=" * 70)

        return "Document Added"


if __name__ == '__main__' :
    DocumentProcessor().document_processing_pipeline_with_track('docs/giddiness.pdf', Embedding().get_embedding_model())

