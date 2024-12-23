"""
Utility functions for working with LangChain prompts and documents.

This module provides helper functions for creating chat prompt templates
and calculating document lengths using LangChain components.
"""

from typing import TypedDict, List, Dict
import json

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI

class PromptTemplateInput(TypedDict):
    """Type definition for prompt template input parameters.

    Attributes:
        system: The system message to set context and instructions
        user: The user message template that can contain input variables
    """
    system: str
    user: str

def get_prompt_template(prompt: PromptTemplateInput) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate with system and user prompts.

    Args:
        prompt: Dictionary containing system and user prompt templates

    Returns:
        A configured chat prompt template ready for use

    Example:
        >>> prompt_input = {
        ...     "system": "You are a helpful assistant.",
        ...     "user": "Please help me with {task}."
        ... }
        >>> template = get_prompt_template(prompt_input)
    """
    # Create message templates
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt["system"])
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt["user"])
    
    # Combine into chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt

def length_function(llm: ChatOpenAI, documents: List[Document]) -> int:
    """Calculate the total number of tokens in a list of documents.

    Args:
        llm: ChatOpenAI instance used for token counting
        documents: List of Document objects to process

    Returns:
        Total number of tokens across all documents
    """
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the parsed JSON data

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON
        FileNotFoundError: If the file doesn't exist
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict, file_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file

    Raises:
        IOError: If there's an error writing the file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)