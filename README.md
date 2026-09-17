# Intelligent Document Query System (RAG)

A 100% free, local Python Retrieval-Augmented Generation (RAG) pipeline designed to execute fast semantic queries across unstructured technical PDF documentation.

## Features
- **Local PDF Ingestion**: Dynamically chunks technical documents into 1000-character segments.
- **Free Vector Embeddings**: Utilizes Hugging Face's `all-MiniLM-L6-v2` via `sentence-transformers` for dense vector generation.
- **Local Vector Database**: Stores embeddings locally using ChromaDB for sub-200ms semantic search retrieval times.
- **Offline LLM**: Integrates with [Ollama](https://ollama.com/) (using `llama3.2`) for completely private, offline, and free query generation.

## Prerequisites
- Python 3
- [Ollama](https://ollama.com/) installed and running locally.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull the Local Model**
   In a separate terminal, pull and run the Llama 3.2 model via Ollama:
   ```bash
   ollama run llama3.2
   ```
   *Note: Keep this terminal open/running in the background.*

## Usage

1. **Add Your Data**
   Place your unstructured technical PDF documents into the `data/` directory.

2. **Ingest Documents**
   Run the ingestion script to parse the PDFs, generate embeddings, and build the local Chroma database:
   ```bash
   python ingest.py
   ```

3. **Query the System**
   Execute a semantic search query against your documents:
   ```bash
   python query.py "What is the recommended operating temperature for the AX-400 controller?"
   ```

## Architecture
- **LangChain**: Orchestrates the document loading, splitting, and RAG retrieval chain.
- **ChromaDB**: Local vector store for the document embeddings.
- **Hugging Face**: Provides the free, local embedding model (`all-MiniLM-L6-v2`).
- **Ollama**: Provides the free, local Large Language Model (`llama3.2`).
