"""
Streamlit app for generating flashcards from documents.

This script provides a web interface for uploading documents,
configuring prompts, and generating flashcards.

Run with: streamlit run app.py
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple
import asyncio

import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from ai_flashcard_generator.flashcard_generator import FlashcardGenerator


def load_default_prompts() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load default prompt configurations.

    Returns:
        Tuple containing map and reduce prompt configurations
    """
    map_prompt = {
        "system": "You are a helpful assistant specialized in effectively summarizing any kind of text",
        "user": """Based on the provided documents, please write a summary by picking out the major CONCEPTS, TERMS, DEFINITIONS,
and ACRONYMS that are important in the documents.

Prioritize clarity and brevity while retaining the essential information.

Aim to convey any supporting details that contribute to a comprehensive understanding of each CONCEPT, TERM, DEFINITION and ACRONYM. 

Do not focus on historical context (when something was introduced or implemented). Ignore anything that looks like source code.

DOCUMENTS:
{docs}

Helpful Answer:
"""
    }

    reduce_prompt = {
        "system": "You are a helpful assistant",
        "user": """The following is set of definitions/concepts:
{docs}
Take these and distill it into a final, consolidated list of at least twenty (20) definitions/concepts.

For each of these, generate a question and an answer. The goal is that these tuples of questions and answers will
be used to create flashcards.

Please provide the result in a JSON format, using questions as keys and answers as values.

Helpful Answer:"""
    }

    return map_prompt, reduce_prompt


def save_flashcards(flashcards: Dict) -> None:
    """Save flashcards and provide download link.

    Args:
        flashcards: Dictionary containing flashcard data
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        json.dump(flashcards, tmp_file, indent=2, ensure_ascii=False)
    
    # Read the file for download
    with open(tmp_file.name, 'r', encoding='utf-8') as f:
        st.download_button(
            label="Download Flashcards",
            data=f,
            file_name="flashcards.json",
            mime="application/json"
        )


def display_flashcards(flashcards: Dict) -> None:
    """Display flashcards in an interactive format.

    Args:
        flashcards: Dictionary containing flashcard data
    """
    st.subheader("Generated Flashcards")
    
    for i, (question, answer) in enumerate(flashcards.items(), 1):
        with st.expander(f"Flashcard {i}: {question}"):
            st.write(answer)


def main():
    """Main Streamlit app function."""
    st.set_page_config(
        page_title="Flashcard Generator",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Document Flashcard Generator")
    st.write("""
    Upload a PDF document and generate flashcards automatically.
    Customize the prompts used for generation if needed.
    """)

    # Sidebar for prompt configuration
    st.sidebar.title("⚙️ Configuration")
    show_advanced = st.sidebar.checkbox("Show Advanced Configuration")

    # Load default prompts
    default_map_prompt, default_reduce_prompt = load_default_prompts()

    # Advanced configuration section
    if show_advanced:
        st.sidebar.subheader("Map Prompt Configuration")
        map_system = st.sidebar.text_area(
            "System Prompt",
            default_map_prompt["system"],
            key="map_system"
        )
        map_user = st.sidebar.text_area(
            "User Prompt",
            default_map_prompt["user"],
            key="map_user",
            height=200
        )
        
        st.sidebar.subheader("Reduce Prompt Configuration")
        reduce_system = st.sidebar.text_area(
            "System Prompt",
            default_reduce_prompt["system"],
            key="reduce_system"
        )
        reduce_user = st.sidebar.text_area(
            "User Prompt",
            default_reduce_prompt["user"],
            key="reduce_user",
            height=200
        )
        
        map_prompt = {"system": map_system, "user": map_user}
        reduce_prompt = {"system": reduce_system, "user": reduce_user}
    else:
        map_prompt = default_map_prompt
        reduce_prompt = default_reduce_prompt

    # Main content area
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Select a PDF file to generate flashcards from"
    )

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        if st.button("Generate Flashcards", type="primary"):
            with st.spinner("Generating flashcards..."):
                try:
                    # Initialize generator
                    generator = FlashcardGenerator(
                        file_path=temp_path,
                        map_prompt=map_prompt,
                        reduce_prompt=reduce_prompt,
                        map_parser=StrOutputParser(),
                        reduce_parser=JsonOutputParser()
                    )
                    
                    # Generate flashcards
                    flashcards = asyncio.run(generator.create_flashcards())
                    
                    # Display success message
                    st.success("Flashcards generated successfully!")
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Interactive View", "Raw JSON"])
                    
                    with tab1:
                        display_flashcards(flashcards)
                    
                    with tab2:
                        st.json(flashcards)
                        save_flashcards(flashcards)
                
                except Exception as e:
                    st.error(f"Error generating flashcards: {str(e)}")
                
                finally:
                    # Cleanup temporary file
                    Path(temp_path).unlink()

    # Footer
    st.divider()
    st.markdown("""
    ### 📝 Instructions
    1. Upload a PDF document using the file uploader above
    2. (Optional) Customize the generation prompts in the sidebar
    3. Click "Generate Flashcards" to process your document
    4. View the results and download the generated flashcards
    """)


if __name__ == "__main__":
    main()