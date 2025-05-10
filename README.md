# RAG-Powered Multi-Agent Q&A System

This project implements a question-answering system that combines Retrieval-Augmented Generation (RAG) with a multi-agent approach to handle different types of queries.

# Features

- Document Ingestion: Upload PDF and TXT files to build a knowledge base
- Vector Embeddings: Creates semantic vector embeddings for efficient retrieval
- Multi-Agent Routing:
  - Calculator for mathematical queries
  - Dictionary for definition requests
  - RAG pipeline for knowledge-based questions
- Decision Logging: Tracks the system's decision process
- Interactive UI: Simple Streamlit interface to interact with the system

# Architecture

The system is composed of the following components:

1. Document Processing:
   - Ingests PDF and TXT files
   - Chunks documents into manageable pieces
   - Creates vector embeddings using OpenAI's embedding model
   - Stores embeddings in a FAISS vector index

2. Query Processing:
   - Analyzes query intent using keyword detection
   - Routes to appropriate agent (Calculator, Dictionary, or RAG)
   - Retrieves relevant context for RAG queries
   - Generates responses using OpenAI's language model

3. User Interface:
   - Document upload and management
   - Query input
   - Results display (agent used, context, answer)
   - Process logs

# Requirements

- Python 3.7+
- OpenAI API key
- Required Python packages (see below)

# Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   pip install streamlit langchain langchain_openai langchain_community faiss-cpu pypdf2 simpleeval
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

# Usage

1. Start the application:
   ```
   streamlit run app.py
   ```
2. Upload one or more documents (PDF or TXT files)
3. Enter a question in the text area
4. Submit your question and view the results

## Query Types

- Calculation queries: Include keywords like "calculate", "compute", "sum", etc.
- Definition queries: Include keywords like "define", "what is", "meaning of", etc.
- Knowledge queries: Any other queries will use the RAG pipeline to find information in your documents

# Example Queries

- "Calculate the square root of 144"
- "Define artificial intelligence"
- "What are the key features of this product?" (assuming product documentation was uploaded)

# Limitations

- The system only supports PDF and TXT files
- Calculator functionality is limited to basic mathematical operations
- RAG retrieval quality depends on the documents provided
