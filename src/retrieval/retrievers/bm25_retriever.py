import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config.settings import settings
from src.indexing.vector_store import VectorStore


class Bm25Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        k: int | None = None,
        min_score: float = 0.0,
    ):
        self.vector_store = vector_store.get_vector_store()
        self.k = settings.TOP_K if k is None else k
        self.min_score = min_score

        self.bm25_retriever = self.build_bm25_from_vectorstore(
            vector_store=self.vector_store,
            k=self.k,
        )

    def build_bm25_from_vectorstore(
        self,
        vector_store,
        k: int | None = None,
    ):
        candidate_k = self.k if k is None else k
        data = vector_store.get(include=["documents", "metadatas"])

        documents = [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(
                data["documents"],
                data["metadatas"],
            )
        ]

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = candidate_k
        return bm25_retriever

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """Return exact top-k indices without sorting the full corpus."""

        if k <= 0 or scores.size == 0:
            return np.array([], dtype=int)

        k = min(k, scores.size)
        if k == scores.size:
            return np.argsort(scores)[::-1]

        partition = np.argpartition(scores, -k)[-k:]
        return partition[np.argsort(scores[partition])[::-1]]

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        candidate_k = self.k if k is None else k

        processed_query = self.bm25_retriever.preprocess_func(query)
        scores = np.asarray(
            self.bm25_retriever.vectorizer.get_scores(processed_query)
        )

        top_indices = self._top_indices(scores, candidate_k)
        results = []

        for index in top_indices:
            score = float(scores[index])
            # A zero score means the lexical branch found no matching evidence.
            # Returning arbitrary zero-score documents gives them undeserved RRF
            # credit and can damage ranking.
            if not np.isfinite(score) or score <= self.min_score:
                continue

            doc = self.bm25_retriever.docs[index]
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "bm25_score": score,
                    "from_bm25": True,
                    "bm25_rank": len(results) + 1,
                }
            )
            results.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        return results

    def get_bm25_retriever(self):
        return self.bm25_retriever
