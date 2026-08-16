from langchain_core.documents import Document
from src.retrieval.retrievers.rrf_fusion import RRFFusion
from src.config.settings import settings
from src.retrieval.query.query import Query


class HybridRetriever:

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        top_k: int = settings.TOP_K,
        rrf_k: int = settings.RRF_K
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

        self.rrf = RRFFusion(
            rrf_k=rrf_k
        )

    def retrieve(self, query: Query) -> list[Document]:
        query = query.normalized_query
        # 1. Vector retrieval
        vector_docs = self.vector_retriever.retrieve(query)

        # 2. BM25 retrieval
        bm25_docs = self.bm25_retriever.retrieve(query)

        # 3. RRF Fusion
        fused_docs = self.rrf.fuse(
            ranked_lists=[
                vector_docs,
                bm25_docs
            ]
        )

        # 4. Deduplication
        unique_docs = self._deduplicate(
            fused_docs
        )

        return unique_docs[:self.top_k]

    def _deduplicate(
        self,
        documents: list[Document]
    ) -> list[Document]:

        seen = set()
        unique_documents = []

        for doc in documents:

            doc_id = self._get_document_id(doc)

            if doc_id in seen:
                continue

            seen.add(doc_id)
            unique_documents.append(doc)

        return unique_documents

    @staticmethod
    def _get_document_id(doc: Document) -> str:

        # Prefer chunk_id if available
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        # Fallback to source + page + content
        return (
            f"{doc.metadata.get('source', '')}:"
            f"{doc.metadata.get('page_number', '')}:"
            f"{doc.page_content}"
        )