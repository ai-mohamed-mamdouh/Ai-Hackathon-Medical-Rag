from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseLoader(ABC):
    """Base interface for all document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        raise NotImplementedError