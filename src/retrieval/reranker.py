import torch
from src.retrieval.query.query import Query
from src.config.settings import settings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class RerankerModel:

    def __init__(self, model_name:str = settings.RERANKER_MODEL_NAME) :
        self.model = CrossEncoder( 
            model_name_or_path=model_name,
            activation_fn=torch.nn.Sigmoid()
            )

    def get_reranker_model(self):
        return self.model

class Reranker:

    def __init__(self, model, top_k: int = settings.TOP_K):
        self.model = model
        self.top_k = top_k

    def rerank(self, query: Query, documents: list[Document] ) -> list[Document]:

        query = query.normalized_query

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(
            pairs
        )

        for doc, score in zip(documents, scores):

            doc.metadata["rerank_score"] = float(score)

        ranked_documents = sorted(
            documents,
            key=lambda doc: doc.metadata["rerank_score"],
            reverse=True
        )

        return ranked_documents[:self.top_k]
