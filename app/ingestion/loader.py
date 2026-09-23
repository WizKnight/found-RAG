from abc import ABC, abstractmethod
from pathlib import Path


class BaseDocumentLoader(ABC):
    """Base interface for document loaders."""

    @abstractmethod
    def load(self, path: Path) -> list[str]:
        """Load a document and return page/section-level text."""
        raise NotImplementedError