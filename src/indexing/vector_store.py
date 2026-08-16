import hashlib
from src.config.settings import settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.collection_name = settings.COLLECTION_NAME
        self.vector_store = self.get_vector_store()

    def get_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db",
        )

    def generate_chunks_ids(
        self,
        chunks: list[Document],
    ) -> list[str]:
        """
        Input:
            A list of LangChain Document objects representing the final chunks.

        Output:
            A list of deterministic unique IDs, one ID for each chunk.

            Each ID is generated using SHA-256 from the chunk's source,
            page, section, and page_content. The same chunk will always
            generate the same ID.
        """

        ids = []

        for chunk in chunks:
            value = (
                f"{chunk.metadata.get('source', '')}|"
                f"{chunk.metadata.get('page', '')}|"
                f"{chunk.metadata.get('section', '')}|"
                f"{chunk.page_content}"
            )

            chunk_id = hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        return ids

    def add_new_documents(
        self,
        vector_store: Chroma,
        chunks: list[Document],
        ids: list[str],
    ) -> list[str]:
        """
        Input:
            vector_store:
                A configured LangChain Chroma vector store.

            chunks:
                A list of final LangChain Document chunks.

            ids:
                A list of deterministic IDs corresponding to the chunks.

        Output:
            A list of IDs that were actually added to the vector store.

        The function checks which chunk IDs already exist in the vector store
        and adds only new chunks, preventing duplicate storage and unnecessary
        embedding computation.
        """

        if len(chunks) != len(ids):
            raise ValueError(
                "The number of chunks must match the number of IDs."
            )

        existing_documents = vector_store.get_by_ids(ids)

        existing_ids = {
            document.id
            for document in existing_documents
            if document.id is not None
        }

        new_chunks = []
        new_ids = []

        for chunk, chunk_id in zip(chunks, ids):
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if not new_chunks:
            return []

        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

        return new_ids
    