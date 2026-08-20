from time import perf_counter

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.retrieval.query.query import Query


class RerankerModel:
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        model_kwargs = {
            "model_name_or_path": model_name,
            "activation_fn": torch.nn.Identity(),
        }

        device = getattr(settings, "RERANKER_DEVICE", None)
        if device:
            model_kwargs["device"] = device

        max_length = getattr(settings, "RERANKER_MAX_LENGTH", None)
        if max_length:
            model_kwargs["max_length"] = max_length

        start = perf_counter()
        self.model = CrossEncoder(**model_kwargs)
        self.load_ms = (perf_counter() - start) * 1000

    def get_reranker_model(self):
        return self.model


class Reranker:
    def __init__(
        self,
        model,
        top_k: int = settings.TOP_K,
        batch_size: int | None = None,
    ):
        self.model_load_ms = getattr(model, "load_ms", None)
        if hasattr(model, "get_reranker_model"):
            model = model.get_reranker_model()

        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size

    def rerank(
        self,
        query: Query,
        documents: list[Document],
    ) -> list[Document]:
        normalized_query = query.normalized_query

        if not documents:
            return []

        pairs = [
            (normalized_query, doc.page_content)
            for doc in documents
        ]

        predict_kwargs = {"show_progress_bar": False}
        if self.batch_size is not None:
            predict_kwargs["batch_size"] = self.batch_size

        scores = self.model.predict(pairs, **predict_kwargs)

        ranked_documents = []
        for doc, score in zip(documents, scores):
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = float(score)
            ranked_documents.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        ranked_documents.sort(
            key=lambda doc: doc.metadata["rerank_score"],
            reverse=True,
        )
        return ranked_documents[: self.top_k]
