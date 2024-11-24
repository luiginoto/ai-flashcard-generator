import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

TOKEN_MAX = 100000

# Load environment variables from .env file if it exists
load_dotenv()

def get_llm():
    """
    Returns an instance of the language model configured with specified settings.
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=2
    )