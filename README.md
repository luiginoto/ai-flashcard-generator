# 📚 AI Flashcard Generator

## 🌟 Overview

AI Flashcard Generator is an intelligent tool that transforms PDF documents into interactive learning flashcards using large language models and map-reduce processing.

![Demo GIF](path/to/demo.gif)  # Replace with an actual project demo gif if available

## ✨ Features

- 🤖 AI-powered content extraction
- 📄 PDF document processing
- 🃏 Automatic flashcard generation
- 🌐 Web and CLI interfaces
- 🔬 Customizable generation prompts

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [License](#-license)

## 🔧 Prerequisites

Before getting started, ensure you have the following:

### System Requirements
- Python 3.11+
- pip
- make
- virtualenv (recommended)

### API and Accounts
- OpenAI API Key (sign up at [OpenAI Platform](https://platform.openai.com/))
    1. Create an OpenAI account
    2. Generate an API key from your OpenAI account settings
    3. Have your API key ready for configuration

## 💾 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-flashcard-generator.git
cd ai-flashcard-generator
```

### 2. API Key Setup

To use the AI Flashcard Generator, you'll need to set up your OpenAI API key in a `.env` file. Follow these steps:

1. Create a `.env` file in the root directory of the project:

```bash
touch .env
```

2. Open the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

#### Important Security Notes
- Replace `your_actual_openai_api_key_here` with your real OpenAI API key
- Never commit your `.env` file to version control
- Add `.env` to your `.gitignore` file to prevent accidental exposure

### 3. Set Up Environment

The project uses a Makefile to simplify environment setup:

```bash
# Create virtual environment and install dependencies
make create_environment

# Install project requirements
make requirements
```

#### Alternative Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Development Workflow

Common Makefile commands:
- `make environment`: Set up virtual environment
- `make requirements`: Install dependencies
- `make lint`: Run linters
- `make format`: Run source code formatting
- `make clean`: Clean up temporary files

## 🚀 Usage Methods

### 1. Command Line Interface

Generate flashcards directly from the terminal:

```bash
python main.py --input path/to/your/document.pdf \
               --output flashcards.json \
               --map-prompt config/map_prompt.json \
               --reduce-prompt config/reduce_prompt.json
```

#### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input PDF path | Required |
| `--output` | `-o` | Output JSON path | Required |
| `--map-prompt` | `-m` | Map prompt config | `config/map_prompt.json` |
| `--reduce-prompt` | `-r` | Reduce prompt config | `config/reduce_prompt.json` |

### 2. Streamlit Web Interface

Launch the interactive web application:

```bash
streamlit run app.py
```

#### Web App Features:
- Upload PDF documents
- Customize generation prompts
- Interactive flashcard preview
- Download generated flashcards
- JSON export option

## How It Works

The AI Flashcard Generator uses a sophisticated map-reduce workflow:

1. **Document Loading**: PDF is loaded and split into manageable chunks
2. **Mapping Phase**: Each document chunk is processed to extract key information
3. **Reduction Phase**: Extracted information is consolidated into flashcards
4. **Output**: Flashcards are saved as a JSON file

### Key Components

- `DocumentLoader`: Handles PDF parsing and text splitting
- `MapReduceGraph`: Orchestrates the AI-powered processing workflow
- `FlashcardGenerator`: Coordinates the entire flashcard generation process
- `Streamlit App`: Provides a user-friendly web interface

## 🛠 Configuration

### Prompt Customization

You can customize generation prompts in `config/map_prompt.json` and `config/reduce_prompt.json`:

```json
{
    "system": "You are a specialized assistant",
    "user": "Extract key concepts from the text..."
}
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
