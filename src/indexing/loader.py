from langchain_core.documents import Document
from src.indexing.loaders.base import BaseLoader

class DocumentLoader :
    def load_data(self, loader: BaseLoader) -> list[Document]:
        """
        Load documents using the provided loader.

        Args:
            loader: An initialized BaseLoader implementation.

        Returns:
            A list of LangChain Document objects.
        """
        return loader.load()