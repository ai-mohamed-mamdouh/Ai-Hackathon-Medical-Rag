from langchain_core.documents import Document

from src.config.settings import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.query.query import Query, QueryProcessor
from src.retrieval.reranker import Reranker, RerankerModel
from src.retrieval.retrievers.bm25_retriever import Bm25Retriever
from src.retrieval.retrievers.vector_retriever import VectorRetriever
from src.retrieval.retrievers.hybrid_retriever import HybridRetriever


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.query_processor = QueryProcessor()

        self.vector_retriever = VectorRetriever(
            vector_store=vector_store
        )

        self.bm25_retriever = Bm25Retriever(
            vector_store=vector_store
        )

        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=settings.TOP_K
        )

    def threshold(
        self,
        documents: list[Document]
    ) -> list[Document]:

        return [
            doc
            for doc in documents
            if doc.metadata.get("rerank_score", 0)
            > settings.RELEVANCE_THRESHOLD
        ]

    def _retrieve_single_query(self, query: Query, reranker: Reranker) -> list[Document]:

        if not query.normalized_query:
            query = self.query_processor.normalize_query(
                query=query
            )

        hybrid_docs = self.hybrid_retriever.retrieve(
            query=query
        )

        reranked_docs = reranker.rerank(
            query=query,
            documents=hybrid_docs
        )

        return self.threshold(reranked_docs)

    def retrieval_pipeline(self, query: Query, reranker_model: RerankerModel, decomposition: bool = False) -> list[list[Document]]:

        reranker = Reranker(
            model=reranker_model
        )

        if decomposition:
            queries = self.query_processor.decompose_query(
                query=query
            )
        else:
            queries = [query]

        documents_for_queries: list[list[Document]] = []

        for current_query in queries:
            documents = self._retrieve_single_query(
                query=current_query,
                reranker=reranker
            )

            documents_for_queries.append(documents)

        print(len(documents_for_queries))
        for q in queries :
            print(q.normalized_query)
            print('==============')

        return documents_for_queries

    