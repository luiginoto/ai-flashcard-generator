"""
Module for loading and processing PDF documents.

This module provides functionality for loading PDF documents and splitting them into
manageable chunks using the LangChain document processing pipeline.
"""

from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ai_flashcard_generator.base_classes import BaseDocumentLoader


class DocumentLoader(BaseDocumentLoader):

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 0
    ) -> None:
        """
        Initialize the DocumentLoader with file path and chunking parameters.

        Args:
            file_path (str): Path to the PDF file to be processed.
            chunk_size (int, optional): The target size of each text chunk in tokens.
                Defaults to 1000.
            chunk_overlap (int, optional): The number of tokens of overlap between
                chunks. Defaults to 0.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self) -> List[Document]:
        """
        Load a PDF document and split it into chunks.

        This method loads the PDF file specified in the file_path and splits its
        content into chunks using tiktoken-based text splitting. The splitting
        process considers both the chunk size and overlap parameters specified
        during initialization.

        Returns:
            List[Document]: A list of Document objects, each containing a chunk of
                the original PDF's content along with metadata.

        Raises:
            FileNotFoundError: If the specified PDF file cannot be found.
            ValueError: If the PDF file is corrupted or cannot be processed.

        Example:
            >>> chunks = loader.load_and_split()
            >>> for chunk in chunks:
            ...     print(f"Chunk content length: {len(chunk.page_content)}")
        """
        loader = PyPDFLoader(self.file_path)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return loader.load_and_split(text_splitter=text_splitter)