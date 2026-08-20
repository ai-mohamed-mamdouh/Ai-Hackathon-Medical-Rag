from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from langchain_core.documents import Document

from src.config.settings import settings
from src.retrieval.debug import stage_snapshot
from src.retrieval.query.query import Query
from src.retrieval.retrievers.document_id import get_document_id
from src.retrieval.retrievers.rrf_fusion import RRFFusion


class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        top_k: int = settings.TOP_K,
        rrf_k: int = settings.RRF_K,
        vector_k: int | None = None,
        bm25_k: int | None = None,
        parallel_retrieval: bool = False,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.parallel_retrieval = parallel_retrieval
        self.rrf = RRFFusion(rrf_k=rrf_k)

    @staticmethod
    def _timed_call(func, *args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        return result, (perf_counter() - start) * 1000

    def _retrieve_branches(self, query: str):
        if not self.parallel_retrieval:
            vector_docs, vector_ms = self._timed_call(
                self.vector_retriever.retrieve,
                query,
                self.vector_k,
            )
            bm25_docs, bm25_ms = self._timed_call(
                self.bm25_retriever.retrieve,
                query,
                self.bm25_k,
            )
            return vector_docs, bm25_docs, vector_ms, bm25_ms

        # Opt-in only: the vector client implementation is outside this folder,
        # so thread safety must be verified in the deployment before enabling.
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(
                self._timed_call,
                self.vector_retriever.retrieve,
                query,
                self.vector_k,
            )
            bm25_future = executor.submit(
                self._timed_call,
                self.bm25_retriever.retrieve,
                query,
                self.bm25_k,
            )
            vector_docs, vector_ms = vector_future.result()
            bm25_docs, bm25_ms = bm25_future.result()

        return vector_docs, bm25_docs, vector_ms, bm25_ms

    def retrieve(
        self,
        query: Query,
        trace: dict | None = None,
    ) -> list[Document]:
        normalized_query = query.normalized_query
        vector_docs, bm25_docs, vector_ms, bm25_ms = self._retrieve_branches(
            normalized_query
        )

        rrf_start = perf_counter()
        fused_docs = self.rrf.fuse(
            ranked_lists=[vector_docs, bm25_docs]
        )
        rrf_ms = (perf_counter() - rrf_start) * 1000

        dedup_start = perf_counter()
        unique_docs = self._deduplicate(fused_docs)
        dedup_ms = (perf_counter() - dedup_start) * 1000

        candidates = unique_docs[: self.top_k]

        if trace is not None:
            trace["timings"].update(
                {
                    "vector_search_ms": vector_ms,
                    "bm25_search_ms": bm25_ms,
                    "rrf_ms": rrf_ms,
                    "deduplication_ms": dedup_ms,
                }
            )
            trace["stages"].update(
                {
                    "vector": stage_snapshot(vector_docs),
                    "bm25": stage_snapshot(bm25_docs),
                    "rrf": stage_snapshot(fused_docs),
                    "after_dedup": stage_snapshot(unique_docs),
                    "reranker_input": stage_snapshot(candidates),
                }
            )
            trace["candidate_funnel"].update(
                {
                    "vector": len(vector_docs),
                    "bm25": len(bm25_docs),
                    "after_union_rrf": len(fused_docs),
                    "after_dedup": len(unique_docs),
                    "reranker_input": len(candidates),
                }
            )

        return candidates

    def _deduplicate(self, documents: list[Document]) -> list[Document]:
        seen = set()
        unique_documents = []

        for doc in documents:
            doc_id = get_document_id(doc)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique_documents.append(doc)

        return unique_documents
