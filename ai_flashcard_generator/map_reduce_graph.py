"""
Module implementing a map-reduce graph for document processing.

This module provides a graph-based implementation of the map-reduce pattern,
specifically designed for processing and summarizing documents using LangChain
components.
"""

from typing import List, Annotated, TypedDict, Callable, Literal, Any
import operator

from langchain_core.documents import Document
from langchain_core.runnables.base import RunnableSequence
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from base_classes import BaseGraph
from config import TOKEN_MAX


class MapReduceGraph(BaseGraph):
    """
    A graph implementation for map-reduce operations on documents.

    This class implements a graph-based approach to process documents using a
    map-reduce pattern. It handles document summarization through parallel processing
    (map) and subsequent combination of results (reduce).

    Attributes:
        graph (StateGraph): The underlying state graph instance.
        map_chain (RunnableSequence): Chain for mapping operations.
        reduce_chain (RunnableSequence): Chain for reduction operations.
        length_function (Callable): Function to calculate document length.
        app: Compiled graph application.
    """

    class OverallState(TypedDict):
        """State definition for the overall graph processing."""
        contents: List[str]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str

    class SummaryState(TypedDict):
        """State definition for individual summary operations."""
        content: str

    def __init__(
        self,
        map_chain: RunnableSequence,
        reduce_chain: RunnableSequence,
        length_function: Callable[[List[Document]], int]
    ) -> None:
        """
        Initialize the MapReduceGraph.

        Args:
            map_chain (RunnableSequence): Chain for mapping operations.
            reduce_chain (RunnableSequence): Chain for reduction operations.
            length_function (Callable[[List[Document]], int]): Function to calculate
                document length.
        """
        self.graph = StateGraph(self.OverallState)
        self.map_chain = map_chain
        self.reduce_chain = reduce_chain
        self.length_function = length_function
        self.app = None

    def add_nodes(self) -> None:
        """
        Add processing nodes to the state graph.

        Adds all necessary nodes for the map-reduce workflow:
        - Summary generation
        - Summary collection
        - Summary collapse
        - Final summary generation
        """
        self.graph.add_node("generate_summary", self._generate_summary)
        self.graph.add_node("collect_summaries", self._collect_summaries)
        self.graph.add_node("collapse_summaries", self._collapse_summaries)
        self.graph.add_node("generate_final_summary", self._generate_final_summary)

    def add_edges(self) -> None:
        """
        Add edges between nodes in the state graph.

        Configures the workflow by connecting nodes with appropriate edges and
        conditions for the map-reduce process.
        """
        self.graph.add_conditional_edges(START, self._map_summaries, ["generate_summary"])
        self.graph.add_edge("generate_summary", "collect_summaries")
        self.graph.add_conditional_edges("collect_summaries", self._should_collapse)
        self.graph.add_conditional_edges("collapse_summaries", self._should_collapse)
        self.graph.add_edge("generate_final_summary", END)
    
    def build(self) -> None:
        """
        Build and compile the graph.

        Adds all nodes and edges, then compiles the graph into an executable
        application.
        """
        self.add_nodes()
        self.add_edges()
        self.app = self.graph.compile()
    
    async def get_result(self, input_contents: List[str]) -> Any:
        """
        Process input contents and return the final summary.

        Args:
            input_contents (List[str]): List of content strings to process.

        Returns:
            Any: The final generated summary in any format.
        """
        result = await self.app.ainvoke({"contents": input_contents})
        return result["final_summary"]

    def _map_summaries(self, state: OverallState) -> List[Send]:
        """
        Map input contents to summary generation tasks.

        Args:
            state (OverallState): Current state containing input contents.

        Returns:
            List[Send]: List of Send objects for summary generation.
        """
        return [Send("generate_summary", {"content": content}) for content in state["contents"]]

    async def _generate_summary(self, state: SummaryState) -> dict:
        """
        Generate a summary for a single piece of content.

        Args:
            state (SummaryState): State containing the content to summarize.

        Returns:
            dict: Dictionary containing the generated summary.
        """
        response = await self.map_chain.ainvoke(state["content"])
        return {"summaries": [response]}
    
    def _collect_summaries(self, state: OverallState):
        """
        Collect and convert summaries into Document objects.

        Args:
            state (OverallState): Current state containing generated summaries.

        Returns:
            dict: Dictionary containing collapsed summaries as Documents.
        """
        return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}
    
    async def _collapse_summaries(self, state: OverallState) -> dict:
        """
        Collapse multiple summaries into a smaller set of summaries.

        Args:
            state (OverallState): Current state containing summaries to collapse.

        Returns:
            dict: Dictionary containing the collapsed summaries.
        """
        doc_lists = split_list_of_docs(state["collapsed_summaries"], self.length_function, TOKEN_MAX)
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.map_chain.ainvoke))
        return {"collapsed_summaries": results}

    async def _generate_final_summary(self, state) -> dict:
        """
        Generate the final summary from collapsed summaries.

        Args:
            state (OverallState): Current state containing collapsed summaries.

        Returns:
            dict: Dictionary containing the final summary.
        """
        response = await self.reduce_chain.ainvoke(state["summaries"])
        return {"final_summary": response}

    def _should_collapse(
        self,
        state: OverallState
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        """
        Determine whether to continue collapsing summaries or generate final summary.

        Args:
            state (OverallState): Current state containing collapsed summaries.

        Returns:
            Literal["collapse_summaries", "generate_final_summary"]: Next step in the
                process.
        """
        num_tokens = self.length_function(state["collapsed_summaries"])
        return "collapse_summaries" if num_tokens > TOKEN_MAX else "generate_final_summary"