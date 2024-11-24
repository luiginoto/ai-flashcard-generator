from typing import Optional, Any, TypedDict, List

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI

class PromptTemplateInput(TypedDict):
    system: str
    user: str

def get_prompt_template(
    prompt: PromptTemplateInput,
) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate with system and user prompts, and an optional output parser.
    
    Args:
        system_prompt (str): The system message to set context and instructions
        user_prompt (str): The user message template that can contain input variables
        output_parser (Optional[Any]): Parser to structure the output format
    
    Returns:
        ChatPromptTemplate: A configured chat prompt template
    
    Example:
        >>> system_prompt = "You are a helpful assistant."
        >>> user_prompt = "Please help me with {task}."
        >>> template = get_prompt_template(system_prompt, user_prompt)
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
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)