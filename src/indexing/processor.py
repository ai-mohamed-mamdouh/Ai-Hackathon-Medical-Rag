import hashlib
from pathlib import Path
from langchain_core.documents import Document
from src.indexing.loaders import (PDFLoader)
from src.indexing.loader import DocumentLoader
from src.indexing.cleaner import DocumentCleaner
from src.indexing.splitter import DocumentSplitter
from src.indexing.embeddings import Embedding
from src.indexing.vector_store import VectorStore
from langchain_chroma import Chroma


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
        final_chunks = DocumentSplitter().add_chunk_indices_ids(final_chunks)

        ids = vector_store.generate_chunks_ids(final_chunks)
        ids = vector_store.add_new_documents(vector_store=vector_store.get_vector_store(), chunks=final_chunks, ids=ids)

        return 'Document Added.'
    
    def document_processing_pipeline_with_track(self, path : str, embedding_model) :
        documents = DocumentLoader().load_data(PDFLoader(path=path))
        print('pdf loaded')
        clean_documents = DocumentCleaner().clean_documents(documents=documents)
        print('pdf clean')
        sections = DocumentSplitter().split_documents_by_headings(clean_documents)
        print('sections')
        blocks = DocumentSplitter().split_text_and_tables(sections)
        print('blocks')
        chunks = DocumentSplitter().split_documents_to_chunks(blocks)
        print('chunks')
        chunks = self.enrich_documents(chunks) 
        print('enrich chunks')
        chunks = self.deduplicate_documents(chunks)
        final_chunks = DocumentSplitter().add_chunk_indices(chunks)
        print('final chunks')

        vector_store = VectorStore(embedding_model=embedding_model)
        ids = vector_store.generate_chunks_ids(final_chunks)
        print('ids')
        ids = vector_store.add_new_documents(vector_store=vector_store.get_vector_store(), chunks=final_chunks, ids=ids)
        print('finish....')

        print('Indexing Donnnne.....')
        print('=====================================================================')
        return 'Document Added'

    def update_document(self, file_id: str, new_file_path: str, vector_store: Chroma,):
        """
        Update an existing document in Chroma.

        Identity:
            file_id    = hash(filename)
            version_id = hash(file bytes)
            chunk_id   = hash(page_content)

        Reconciliation:
            deleted   = old_chunk_ids - new_chunk_ids
            added     = new_chunk_ids - old_chunk_ids
            unchanged = old_chunk_ids & new_chunk_ids
        """

        # =========================================================
        # 1. Validate new file
        # =========================================================

        path = Path(new_file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {new_file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {new_file_path}"
            )

        # =========================================================
        # 2. Calculate NEW version_id
        # =========================================================

        new_version_id = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        # =========================================================
        # 3. Get existing records from Chroma
        # =========================================================
        #
        # بنجيب documents نفسها لأن الـ IDs القديمة عندك
        # معمولة بالطريقة القديمة:
        #
        # source + page + section + content
        #
        # وإحنا دلوقتي عايزين نقارن باستخدام:
        #
        # hash(page_content)
        #
        # =========================================================

        existing_data = vector_store.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        existing_ids = existing_data.get(
            "ids", []
        )

        existing_documents = existing_data.get(
            "documents", []
        )

        existing_metadatas = existing_data.get(
            "metadatas", []
        )

        # =========================================================
        # 4. Find chunks belonging to file_id
        # =========================================================

        old_records = []

        for record_id, document, metadata in zip(
            existing_ids,
            existing_documents,
            existing_metadatas,
        ):

            metadata = metadata or {}

            # -----------------------------------------------------
            # New indexing format:
            # metadata already contains file_id
            # -----------------------------------------------------

            metadata_file_id = metadata.get(
                "file_id"
            )

            if metadata_file_id == file_id:

                old_records.append({
                    "record_id": record_id,
                    "document": document,
                    "metadata": metadata,
                })

                continue

            # -----------------------------------------------------
            # Backward compatibility with your OLD indexing
            #
            # If file_id doesn't exist yet in metadata,
            # calculate it from source filename.
            # -----------------------------------------------------

            source = metadata.get(
                "source"
            )

            if not source:
                continue

            source_filename = Path(
                str(source)
            ).name

            calculated_file_id = hashlib.sha256(
                source_filename.encode("utf-8")
            ).hexdigest()

            if calculated_file_id == file_id:

                old_records.append({
                    "record_id": record_id,
                    "document": document,
                    "metadata": metadata,
                })

        if not old_records:
            raise ValueError(
                f"No indexed document found for file_id: {file_id}"
            )

        # =========================================================
        # 5. Check old version_id
        # =========================================================

        old_versions = {
            record["metadata"].get("version_id")
            for record in old_records
            if record["metadata"].get("version_id")
        }

        # لو كل chunks عندها نفس version_id
        # والنسخة هي نفسها الجديدة
        if (
            len(old_versions) == 1
            and new_version_id in old_versions
        ):
            return {
                "status": "no_change",
                "file_id": file_id,
                "version_id": new_version_id,
                "added": 0,
                "deleted": 0,
                "unchanged": len(old_records),
            }

        # =========================================================
        # 6. Process NEW document
        # =========================================================
        #
        # نفس pipeline القديم بالضبط.
        # =========================================================

        documents = DocumentLoader().load_data(
            PDFLoader(
                path=new_file_path
            )
        )

        clean_documents = (
            DocumentCleaner()
            .clean_documents(
                documents=documents
            )
        )

        sections = (
            DocumentSplitter()
            .split_documents_by_headings(
                clean_documents
            )
        )

        blocks = (
            DocumentSplitter()
            .split_text_and_tables(
                sections
            )
        )

        chunks = (
            DocumentSplitter()
            .split_documents_to_chunks(
                blocks
            )
        )

        chunks = self.enrich_documents(
            chunks
        )

        final_chunks = self.deduplicate_documents(
            chunks
        )

        final_chunks = (
            DocumentSplitter()
            .add_chunk_indices(
                final_chunks
            )
        )

        # =========================================================
        # 7. Safety check
        # =========================================================

        if not final_chunks:
            raise RuntimeError(
                "The updated document produced zero chunks. "
                "Update cancelled. Old chunks were NOT deleted."
            )

        # =========================================================
        # 8. Build OLD chunks map
        # =========================================================
        #
        # مهم:
        # مش بنستخدم Chroma old ID للمقارنة.
        #
        # بنحسب:
        #
        # chunk_id = hash(page_content)
        #
        # حتى لو الـ Chroma ID القديم معمول بالطريقة القديمة.
        # =========================================================

        old_chunks = {}

        for record in old_records:

            content = record["document"]

            if not content:
                continue

            chunk_id = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            old_chunks[chunk_id] = record

        # =========================================================
        # 9. Build NEW chunks map
        # =========================================================

        new_chunks = {}

        for chunk in final_chunks:

            content = chunk.page_content

            if not content or not content.strip():
                continue

            # -----------------------------------------------------
            # Your requested chunk_id
            # -----------------------------------------------------

            chunk_id = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            # -----------------------------------------------------
            # Attach identity metadata
            # -----------------------------------------------------

            chunk.metadata["file_id"] = file_id
            chunk.metadata["version_id"] = new_version_id
            chunk.metadata["chunk_id"] = chunk_id

            new_chunks[chunk_id] = chunk

        if not new_chunks:
            raise RuntimeError(
                "No valid chunks generated from updated document."
            )

        # =========================================================
        # 10. Reconciliation
        # =========================================================

        old_chunk_ids = set(
            old_chunks.keys()
        )

        new_chunk_ids = set(
            new_chunks.keys()
        )

        deleted_chunk_ids = (
            old_chunk_ids
            - new_chunk_ids
        )

        added_chunk_ids = (
            new_chunk_ids
            - old_chunk_ids
        )

        unchanged_chunk_ids = (
            old_chunk_ids
            & new_chunk_ids
        )

        # =========================================================
        # 11. ADD new chunks FIRST
        # =========================================================
        #
        # نضيف الجديد قبل ما نحذف القديم.
        #
        # لو embedding حصل فيه error:
        # القديم يفضل موجود.
        # =========================================================

        added_record_ids = []

        if added_chunk_ids:

            chunks_to_add = []
            ids_to_add = []

            for chunk_id in added_chunk_ids:

                chunk = new_chunks[
                    chunk_id
                ]

                # -------------------------------------------------
                # Chroma storage ID
                #
                # chunk_id نفسه مازال hash(content) فقط.
                #
                # لكن لازم الـ record ID يضم file_id حتى لو نفس
                # النص موجود في guideline تانية.
                # -------------------------------------------------

                record_id = hashlib.sha256(
                    f"{file_id}|{chunk_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()

                chunks_to_add.append(
                    chunk
                )

                ids_to_add.append(
                    record_id
                )

            vector_store.add_documents(
                documents=chunks_to_add,
                ids=ids_to_add,
            )

            added_record_ids = ids_to_add

        # =========================================================
        # 12. Update metadata for UNCHANGED chunks
        # =========================================================
        #
        # مفيش embedding جديد.
        #
        # بنحدث فقط:
        #   version_id
        #   page
        #   section
        #   chunk_index
        #   etc...
        # =========================================================

        if unchanged_chunk_ids:

            update_ids = []
            update_metadatas = []

            for chunk_id in unchanged_chunk_ids:

                old_record = old_chunks[
                    chunk_id
                ]

                new_chunk = new_chunks[
                    chunk_id
                ]

                update_ids.append(
                    old_record["record_id"]
                )

                update_metadatas.append(
                    new_chunk.metadata
                )

            vector_store._collection.update(
                ids=update_ids,
                metadatas=update_metadatas,
            )

        # =========================================================
        # 13. DELETE stale chunks
        # =========================================================

        deleted_record_ids = []

        for chunk_id in deleted_chunk_ids:

            old_record = old_chunks[
                chunk_id
            ]

            deleted_record_ids.append(
                old_record["record_id"]
            )

        if deleted_record_ids:

            vector_store.delete(
                ids=deleted_record_ids
            )

        # =========================================================
        # 14. Return report
        # =========================================================

        return {
            "status": "updated",

            "file_id": file_id,

            "new_version_id": new_version_id,

            "old_chunks": len(
                old_chunk_ids
            ),

            "new_chunks": len(
                new_chunk_ids
            ),

            "added": len(
                added_chunk_ids
            ),

            "deleted": len(
                deleted_chunk_ids
            ),

            "unchanged": len(
                unchanged_chunk_ids
            ),
        }

