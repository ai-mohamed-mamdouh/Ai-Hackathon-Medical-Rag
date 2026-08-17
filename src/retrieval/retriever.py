import time
from typing import Any, Callable

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

    # =========================================================
    # Debug Helpers
    # =========================================================

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

        print("\n" + "-" * 70)
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

    def _print_example_document(
        self,
        documents_for_queries: list[list[Document]]
    ) -> None:

        example_doc = None

        for documents in documents_for_queries:
            if documents:
                example_doc = documents[0]
                break

        if example_doc is None:
            print("\nNo final documents returned.")
            return

        metadata = example_doc.metadata

        print("\n" + "=" * 70)
        print("EXAMPLE FINAL DOCUMENT")
        print("=" * 70)

        print("\n[PAGE CONTENT]")
        print(example_doc.page_content)

        print("\n[IMPORTANT METADATA]")

        important_metadata = {
            "filename": metadata.get("file_name"),
            "page_number": metadata.get("page_number"),
            "section": metadata.get("section"),
            "chunk_index": metadata.get("chunk_index"),
            "total_chunks": metadata.get("total_chunks"),
            "rerank_score": metadata.get("rerank_score"),
        }

        for key, value in important_metadata.items():
            print(f"{key:<16}: {value}")

        print("=" * 70)

    # =========================================================
    # Threshold
    # =========================================================

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

    # =========================================================
    # Normal Single Query Retrieval
    # =========================================================

    def _retrieve_single_query(
        self,
        query: Query,
        reranker: Reranker
    ) -> list[Document]:

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

    # =========================================================
    # Tracked Single Query Retrieval
    # =========================================================

    def _retrieve_single_query_with_track(
        self,
        query: Query,
        reranker: Reranker,
        pipeline_start: float,
        query_index: int
    ) -> list[Document]:

        print("\n" + "=" * 70)
        print(f"QUERY {query_index}")
        print("=" * 70)

        if not query.normalized_query:
            query = self._track_step(
                step_name=f"Query {query_index} - Normalize Query",
                func=lambda: self.query_processor.normalize_query(
                    query=query
                ),
                pipeline_start=pipeline_start
            )

        hybrid_docs = self._track_step(
            step_name=f"Query {query_index} - Hybrid Retrieval",
            func=lambda: self.hybrid_retriever.retrieve(
                query=query
            ),
            pipeline_start=pipeline_start
        )

        reranked_docs = self._track_step(
            step_name=f"Query {query_index} - Rerank Documents",
            func=lambda: reranker.rerank(
                query=query,
                documents=hybrid_docs
            ),
            pipeline_start=pipeline_start
        )

        final_docs = self._track_step(
            step_name=f"Query {query_index} - Relevance Threshold",
            func=lambda: self.threshold(
                documents=reranked_docs
            ),
            pipeline_start=pipeline_start
        )

        return final_docs

    # =========================================================
    # Normal Retrieval Pipeline
    # =========================================================

    def retrieval_pipeline(
        self,
        query: Query,
        reranker_model: RerankerModel,
        decomposition: bool = False
    ) -> tuple[list[list[Document]], list[Query]]:

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

        return documents_for_queries, queries

    # =========================================================
    # Tracked Retrieval Pipeline
    # =========================================================

    def retrieval_pipeline_with_track(
        self,
        query: Query,
        reranker_model: RerankerModel,
        decomposition: bool = False
    ) -> tuple[list[list[Document]], list[Query]]:

        pipeline_start = time.perf_counter()

        print("\n" + "=" * 70)
        print("RETRIEVAL PIPELINE STARTED")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. Initialize reranker
        # -----------------------------------------------------

        reranker = self._track_step(
            step_name="1. Initialize Reranker",
            func=lambda: Reranker(
                model=reranker_model
            ),
            pipeline_start=pipeline_start
        )

        # -----------------------------------------------------
        # 2. Query decomposition
        # -----------------------------------------------------

        if decomposition:
            queries = self._track_step(
                step_name="2. Query Decomposition",
                func=lambda: self.query_processor.decompose_query(
                    query=query
                ),
                pipeline_start=pipeline_start
            )

        else:
            queries = [query]

            print("\n" + "-" * 70)
            print("[SKIP] Query Decomposition")
            print("Queries    : 1")
            print(
                f"Total Time : "
                f"{time.perf_counter() - pipeline_start:.2f} sec"
            )

        # -----------------------------------------------------
        # 3. Retrieval for each query
        # -----------------------------------------------------

        documents_for_queries: list[list[Document]] = []

        for index, current_query in enumerate(
            queries,
            start=1
        ):
            documents = self._retrieve_single_query_with_track(
                query=current_query,
                reranker=reranker,
                pipeline_start=pipeline_start,
                query_index=index
            )

            documents_for_queries.append(documents)

        # -----------------------------------------------------
        # Final statistics
        # -----------------------------------------------------

        total_time = time.perf_counter() - pipeline_start

        total_documents = sum(
            len(documents)
            for documents in documents_for_queries
        )

        print("\n" + "=" * 70)
        print("RETRIEVAL PIPELINE DONE")
        print("=" * 70)

        print(f"Total Time          : {total_time:.2f} sec")
        print(f"Number of Queries   : {len(queries)}")
        print(f"Total Final Docs    : {total_documents}")
        print(
            f"Threshold           : "
            f"{settings.RELEVANCE_THRESHOLD}"
        )
        print(
            f"Output Type         : "
            f"{type(documents_for_queries).__name__}"
        )

        # -----------------------------------------------------
        # Example final document
        # -----------------------------------------------------

        self._print_example_document(
            documents_for_queries=documents_for_queries
        )

        return documents_for_queries, queries


