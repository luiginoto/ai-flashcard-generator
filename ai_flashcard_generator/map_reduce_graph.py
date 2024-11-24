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
        self.graph = StateGraph(self.OverallState)
        self.map_chain = map_chain
        self.reduce_chain = reduce_chain
        self.length_function = length_function
        self.app = None

    def add_nodes(self) -> None:
        self.graph.add_node("generate_summary", self._generate_summary)
        self.graph.add_node("collect_summaries", self._collect_summaries)
        self.graph.add_node("collapse_summaries", self._collapse_summaries)
        self.graph.add_node("generate_final_summary", self._generate_final_summary)

    def add_edges(self) -> None:
        self.graph.add_conditional_edges(START, self._map_summaries, ["generate_summary"])
        self.graph.add_edge("generate_summary", "collect_summaries")
        self.graph.add_conditional_edges("collect_summaries", self._should_collapse)
        self.graph.add_conditional_edges("collapse_summaries", self._should_collapse)
        self.graph.add_edge("generate_final_summary", END)
    
    def build(self) -> None:
        self.add_nodes()
        self.add_edges()
        self.app = self.graph.compile()
    
    async def get_result(self, input_contents: List[str]) -> Any:
        result = await self.app.ainvoke({"contents": input_contents})
        return result["final_summary"]

    def _map_summaries(self, state: OverallState) -> List[Send]:
        return [Send("generate_summary", {"content": content}) for content in state["contents"]]

    async def _generate_summary(self, state: SummaryState) -> dict:
        response = await self.map_chain.ainvoke(state["content"])
        return {"summaries": [response]}
    
    def _collect_summaries(self, state: OverallState):
        return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}
    
    async def _collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(state["collapsed_summaries"], self.length_function, TOKEN_MAX)
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.map_chain.ainvoke))
        return {"collapsed_summaries": results}

    async def _generate_final_summary(self, state):
        response = await self.reduce_chain.ainvoke(state["summaries"])
        return {"final_summary": response}

    def _should_collapse(self, state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self.length_function(state["collapsed_summaries"])
        return "collapse_summaries" if num_tokens > TOKEN_MAX else "generate_final_summary"