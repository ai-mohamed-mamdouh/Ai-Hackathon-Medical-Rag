from src.retrieval.query.query import Query, QueryProcessor
from src.config.settings import settings
from langchain_core.documents import Document
from src.indexing.vector_store import VectorStore
from src.retrieval.retrievers.bm25_retriever import Bm25Retriever
from src.retrieval.retrievers.vector_retriever import VectorRetriever
from src.retrieval.retrievers.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker, RerankerModel


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

        self.vector_retriever = VectorRetriever(
            vector_store=self.vector_store
        )

        self.bm25_retriever = Bm25Retriever(
            vector_store=self.vector_store
        )

        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=settings.TOP_K
        )

    def threshold(self, documents : list[Document]) :
        threshold = settings.RELEVANCE_THRESHOLD

        relevant_docs = []
        for doc in documents:
            if doc.metadata['rerank_score'] > threshold :
                relevant_docs.append(doc)

        if len(relevant_docs) == 0 :
            relevant_docs.append(None)
            
        return relevant_docs
        
    def retrieval_pipeline(self, query: Query, reRanker_model ) -> list[Document]:
        query = QueryProcessor().normalize_query(query=query)

        hybrid_docs = self.hybrid_retriever.retrieve(
            query=query
        )

        reRanker_docs = Reranker(model=reRanker_model).rerank(
            query=query,
            documents=hybrid_docs
            )
        
        relevant_docs = self.threshold(reRanker_docs)

        return relevant_docs
