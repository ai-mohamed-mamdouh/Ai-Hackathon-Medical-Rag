# return embedding model only ... any provider , any model
from src.config.settings import settings
from langchain_huggingface import HuggingFaceEmbeddings

class Embedding :
    def __init__(self) :
        self.embedding_model = self.get_embedding_model()

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,

            model_kwargs={
                "device": settings.EMBEDDING_DEVICE,
            },

            # Documents
            encode_kwargs={
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
                "normalize_embeddings": True,
            },

            # Queries
            query_encode_kwargs={
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
                "normalize_embeddings": True,
                "prompt_name": "query",
            },
        )
        return embeddings

    def embeddingQuery(self, query : str) :
        return self.embedding_model.embed_query(text=query)