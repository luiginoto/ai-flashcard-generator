"""
A module for generating flashcards from document content using LangChain components.

This module provides functionality to process documents and generate flashcards
using a map-reduce approach with language models.
"""

from typing import Dict, Optional, Type

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.base import BaseOutputParser

from config import get_llm
from document_loader import DocumentLoader
from map_reduce_graph import MapReduceGraph
from utils import get_prompt_template, length_function, PromptTemplateInput


class FlashcardGenerator:
    """A class to generate flashcards from documents using LLM-based processing.

    This class implements a map-reduce pattern to process documents and generate
    flashcards using language models. It handles document loading, processing,
    and flashcard generation through a series of configurable prompts and chains.

    Attributes:
        file_path: Path to the input document file
        llm: Language model instance for processing
        documents: List of loaded and split documents
        map_prompt_template: Template for mapping documents to intermediate results
        reduce_prompt_template: Template for reducing intermediate results to flashcards
        map_chain: Processing chain for the mapping phase
        reduce_chain: Processing chain for the reduction phase
        graph: MapReduceGraph instance for orchestrating the process

    Args:
        file_path: String path to the document file to process
        map_prompt: Prompt template input for the mapping phase
        reduce_prompt: Prompt template input for the reduction phase
        map_parser: Output parser class for mapping phase results (default: StrOutputParser)
        reduce_parser: Output parser class for reduction phase results (default: JsonOutputParser)
    """

    def __init__(
        self,
        file_path: str,
        map_prompt: PromptTemplateInput,
        reduce_prompt: PromptTemplateInput,
        map_parser: Type[BaseOutputParser] = StrOutputParser,
        reduce_parser: Type[BaseOutputParser] = JsonOutputParser
    ) -> None:
        self.file_path = file_path
        self.llm = get_llm()
        self.documents = self._load_documents()

        # Initialize prompts and chains
        self.map_prompt_template = get_prompt_template(map_prompt)
        self.reduce_prompt_template = get_prompt_template(reduce_prompt)
        self.map_chain = self.map_prompt_template | self.llm | map_parser
        self.reduce_chain = self.reduce_prompt_template | self.llm | reduce_parser

        # Configure the length function for document processing
        self.length_function = lambda documents: length_function(self.llm, documents)

        # Build the processing graph
        self.graph = MapReduceGraph(self.map_chain, self.reduce_chain, self.length_function)
        self.graph.build()

    def _load_documents(self) -> list:
        """Load and split the input documents.

        Returns:
            A list of processed Document objects ready for flashcard generation.

        Raises:
            FileNotFoundError: If the input file path doesn't exist
            DocumentLoaderError: If there's an error processing the documents
        """
        loader = DocumentLoader(self.file_path)
        return loader.load_and_split()

    async def create_flashcards(self) -> Dict:
        """Generate flashcards from the loaded documents.

        This method processes the loaded documents through the map-reduce graph
        to generate flashcards.

        Returns:
            A dictionary containing the generated flashcards.

        Raises:
            Exception: If there's an error during the flashcard generation process
        """
        contents = [doc.page_content for doc in self.documents]
        return await self.graph.get_result(contents)