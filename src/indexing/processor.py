import hashlib
from langchain_core.documents import Document
from src.indexing.loaders import (PDFLoader)
from src.indexing.loader import DocumentLoader
from src.indexing.cleaner import DocumentCleaner
from src.indexing.splitter import DocumentSplitter
# from src.indexing.embeddings import Embedding
from src.indexing.vector_store import VectorStore


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
        final_chunks = self.deduplicate_documents(chunks)
        print('final chunks')

        vector_store = VectorStore(embedding_model=embedding_model)
        ids = vector_store.generate_chunks_ids(final_chunks)
        print('ids')
        ids = vector_store.add_new_documents(vector_store=vector_store.get_vector_store(), chunks=final_chunks, ids=ids)
        print('finish....')

        print('Indexing Donnnne.....')
        print('=====================================================================')
        return 'Document Added'


if __name__ == '__main__' :
    DocumentProcessor().document_processing_pipeline('docs/MDKLi.pdf')

