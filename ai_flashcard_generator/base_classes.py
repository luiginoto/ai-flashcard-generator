"""
Base classes for document processing and graph operations.

This module provides abstract base classes for document loading and graph processing
operations, establishing a consistent interface for derived implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langgraph.graph import StateGraph
from langchain_core.runnables.base import RunnableSequence

class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loading and chunking operations.

    This class defines the interface for loading documents and splitting them into
    manageable chunks for further processing. Implementations should handle specific
    document types and splitting strategies.

    Example:
        class PDFLoader(BaseDocumentLoader):
            def load_and_split(self) -> List[Dict[str, Any]]:
                # Implementation for PDF loading and chunking
                pass
    """

    @abstractmethod
    def load_and_split(self) -> List[Any]:
        """
        Load a document and split it into chunks.

        Returns:
            List[Any]: A list of document chunks. The specific type of chunks
                      depends on the implementation but typically includes text
                      content and metadata.

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
            FileNotFoundError: If the source document cannot be found.
            ValueError: If the document format is invalid.
        """
        pass


class BaseGraph(ABC):
    """
    Abstract base class defining the interface for graph operations.

    This class provides the foundational structure for creating and managing graphs,
    defining the essential methods that all graph implementations must provide.
    Concrete implementations should inherit from this class and implement all
    abstract methods according to their specific requirements.

    Example:
        class ConcreteGraph(BaseGraph):
            def add_nodes(self) -> None:
                # Add specific nodes to the graph
                pass

            def add_edges(self) -> None:
                # Define connections between nodes
                pass

            def build(self) -> Graph:
                # Compile and return the final graph
                return compiled_graph
    """

    @abstractmethod
    def add_nodes(self) -> None:
        """
        Add nodes to the state graph.

        This method should implement the logic for adding nodes to the graph,
        including any necessary configuration or initialization of the nodes.

        Returns:
            None

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
        """
        pass

    @abstractmethod
    def add_edges(self) -> None:
        """
        Add edges to the state graph.

        This method should implement the logic for adding edges between nodes,
        defining the connections and relationships within the graph structure.

        Returns:
            None

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
        """
        pass

    @abstractmethod
    def build(self) -> None:
        """
        Compile the state graph.

        This method should implement the logic for finalizing the graph construction
        and saving the compiled graph object as an instance attribute.

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
        """
        pass