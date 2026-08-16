# return embedding model only ... any provider , any model
from src.config.settings import settings
from langchain_huggingface import HuggingFaceEmbeddings

class Embedding :
    def __init__(self) :
        self.embedding_model = self.get_embedding_model()

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )

    def embeddingQuery(self, query : str) :
        return self.embedding_model.embed_query(text=query)