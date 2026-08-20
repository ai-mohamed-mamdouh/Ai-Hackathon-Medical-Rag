from time import perf_counter

from langchain_core.documents import Document
from src.retrieval.postprocessing.semantic_deduplicator import (
    SemanticDeduplicator,
)
from src.indexing.vector_store import VectorStore
from src.retrieval.config import RetrievalConfig
from src.retrieval.debug import stage_snapshot
from src.retrieval.query.query import Query, QueryProcessor
from src.retrieval.reranker import Reranker, RerankerModel
from src.retrieval.retrievers.bm25_retriever import Bm25Retriever
from src.retrieval.retrievers.hybrid_retriever import HybridRetriever
from src.retrieval.retrievers.vector_retriever import VectorRetriever


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ):
        self.config = config or RetrievalConfig.from_settings()
        self.query_processor = QueryProcessor()
        self.semantic_deduplicator = SemanticDeduplicator(
            embeddings=vector_store.embedding_model,
            similarity_threshold=self.config.semantic_dedup_threshold,
            lexical_threshold=self.config.lexical_dedup_threshold,
        )

        self.vector_retriever = VectorRetriever(
            vector_store=vector_store,
            k=self.config.vector_candidate_k,
        )
        self.bm25_retriever = Bm25Retriever(
            vector_store=vector_store,
            k=self.config.bm25_candidate_k,
            min_score=self.config.bm25_min_score,
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=self.config.reranker_input_k,
            rrf_k=self.config.rrf_k,
            vector_k=self.config.vector_candidate_k,
            bm25_k=self.config.bm25_candidate_k,
            parallel_retrieval=self.config.parallel_retrieval,
        )

    def threshold(self, documents: list[Document]) -> list[Document]:
        if self.config.relevance_threshold is None:
            return documents
        
        return [
            doc
            for doc in documents
            if doc.metadata.get("rerank_score", 0)
            > self.config.relevance_threshold
        ]

    @staticmethod
    def _empty_trace(query: Query) -> dict:
        return {
            "query": query.original_query,
            "normalized_query": query.normalized_query,
            "timings": {
                "semantic_deduplication_ms": 0.0,
                "query_normalization_ms": 0.0,
                "embedding_ms": None,
                "vector_search_ms": 0.0,
                "bm25_search_ms": 0.0,
                "rrf_ms": 0.0,
                "deduplication_ms": 0.0,
                "reranker_ms": 0.0,
                "threshold_ms": 0.0,
                "context_expansion_ms": 0.0,
                "serialization_ms": None,
                "total_ms": 0.0,
            },
            "stages": {
                "after_semantic_dedup": [],
                "vector": [],
                "bm25": [],
                "rrf": [],
                "after_dedup": [],
                "reranker_input": [],
                "after_rerank": [],
                "after_threshold": [],
                "final": [],
            },
            "candidate_funnel": {},
        }

    def _retrieve_single_query(
        self,
        query: Query,
        reranker: Reranker,
        debug: bool = False,
    ) -> tuple[list[Document], dict | None]:
        total_start = perf_counter()
        trace = self._empty_trace(query) if debug else None

        normalization_start = perf_counter()
        if not query.normalized_query:
            query = self.query_processor.normalize_query(query=query)
        normalization_ms = (perf_counter() - normalization_start) * 1000
        if trace is not None:
            trace["query"] = query.original_query
            trace["normalized_query"] = query.normalized_query
            trace["timings"]["query_normalization_ms"] = normalization_ms

        hybrid_docs = self.hybrid_retriever.retrieve(
            query=query,
            trace=trace,
        )

        rerank_start = perf_counter()
        reranked_docs = reranker.rerank(
            query=query,
            documents=hybrid_docs,
        )
        reranker_ms = (perf_counter() - rerank_start) * 1000

        threshold_start = perf_counter()
        thresholded_docs = self.threshold(reranked_docs)
        threshold_ms = (perf_counter() - threshold_start) * 1000

        semantic_dedup_start = perf_counter()

        deduplicated_docs = self.semantic_deduplicator.deduplicate(
            thresholded_docs
        )

        semantic_dedup_ms = (
            perf_counter() - semantic_dedup_start
        ) * 1000

        final_docs = deduplicated_docs[: self.config.final_top_k]

        if trace is not None:
            trace["timings"].update(
                {
                    "reranker_ms": reranker_ms,
                    "threshold_ms": threshold_ms,
                    "semantic_deduplication_ms": semantic_dedup_ms,
                    "total_ms": (perf_counter() - total_start) * 1000,
                }
            )

            trace["stages"].update(
                {
                    "after_rerank": stage_snapshot(reranked_docs),
                    "after_threshold": stage_snapshot(thresholded_docs),
                    "after_semantic_dedup": stage_snapshot(deduplicated_docs),
                    "final": stage_snapshot(final_docs),
                }
            )

            trace["candidate_funnel"].update(
                {
                    "after_rerank": len(reranked_docs),
                    "after_threshold": len(thresholded_docs),
                    "after_semantic_dedup": len(deduplicated_docs),
                    "final": len(final_docs),
                }
            )

        return final_docs, trace

    def retrieval_pipeline(
        self,
        query: Query,
        reranker_model: RerankerModel,
        decomposition: bool ,
    ) -> tuple[list[list[Document]], list[Query]]:
        reranker = Reranker(
            model=reranker_model,
            top_k=self.config.reranker_output_k,
            batch_size=self.config.reranker_batch_size,
        )

        if decomposition:
            queries = self.query_processor.decompose_query(query=query)
        else:
            queries = [query]

        documents_for_queries: list[list[Document]] = []
        for current_query in queries:
            documents, _ = self._retrieve_single_query(
                query=current_query,
                reranker=reranker,
                debug=False,
            )
            documents_for_queries.append(documents)

        return documents_for_queries, queries

    def retrieval_pipeline_debug(
        self,
        query: Query,
        reranker_model: RerankerModel,
        decomposition: bool = False,
    ) -> tuple[list[list[Document]], list[Query], list[dict]]:
        reranker = Reranker(
            model=reranker_model,
            top_k=self.config.reranker_output_k,
            batch_size=self.config.reranker_batch_size,
        )

        if decomposition:
            queries = self.query_processor.decompose_query(query=query)
        else:
            queries = [query]

        documents_for_queries: list[list[Document]] = []
        traces: list[dict] = []

        for current_query in queries:
            documents, trace = self._retrieve_single_query(
                query=current_query,
                reranker=reranker,
                debug=True,
            )
            documents_for_queries.append(documents)
            traces.append(trace)

        return documents_for_queries, queries, traces
