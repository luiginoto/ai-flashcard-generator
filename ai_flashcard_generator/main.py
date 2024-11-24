"""
Main script for generating flashcards from PDF documents.

This script provides the entry point for the flashcard generation system,
configuring the necessary prompts and running the flashcard generation process.

Example:
    $ python main.py --input path/to/document.pdf --output flashcards.json 
                     --map-prompt map_prompt.json --reduce-prompt reduce_prompt.json
"""

import argparse
import asyncio
import sys
from typing import Dict

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from flashcard_generator import FlashcardGenerator
from utils import load_json_file, save_json_file


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments

    Raises:
        SystemExit: If any of the provided paths are invalid
    """
    parser = argparse.ArgumentParser(
        description="Generate flashcards from a PDF document",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input PDF document"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path where to save the generated flashcards JSON"
    )
    
    parser.add_argument(
        "--map-prompt",
        "-m",
        type=str,
        default="../config/map_prompt.json",
        help="Path to the JSON file containing map prompt configuration"
    )
    
    parser.add_argument(
        "--reduce-prompt",
        "-r",
        type=str,
        default="../config/reduce_prompt.json",
        help="Path to the JSON file containing reduce prompt configuration"
    )
    
    args = parser.parse_args()
    
    return args

async def main() -> None:
    """Main entry point for the flashcard generation script.

    This function configures the flashcard generator with appropriate prompts
    and parsers, then runs the flashcard generation process.

    Raises:
        FileNotFoundError: If any required files cannot be accessed
        json.JSONDecodeError: If prompt files contain invalid JSON
        Exception: If there's an error during flashcard generation
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load prompt configurations
        map_prompt = load_json_file(args.map_prompt)
        reduce_prompt = load_json_file(args.reduce_prompt)
        
        # Initialize generator with prompts and parsers
        flashcard_generator = FlashcardGenerator(
            file_path=args.input,
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
            map_parser=StrOutputParser(),
            reduce_parser=JsonOutputParser()
        )
        
        # Generate flashcards
        flashcards = await flashcard_generator.create_flashcards()
        
        # Save results
        save_json_file(flashcards, args.output)
        print(f"Flashcards successfully saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())