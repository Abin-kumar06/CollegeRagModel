# College Student Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system for college students to query official documents.

## Features
- Ingests PDF, DOCX, HTML, Text documents
- Chunks documents with overlap
- Embeds using Ollama's nomic-embed-text
- Stores in ChromaDB vector database
- Answers questions with citations from retrieved chunks
- Streamlit UI for easy interaction

## Setup
1. Install Ollama and pull models:
   ```
   ollama pull nomic-embed-text
   ollama pull phi3
   ```

2. Install Python dependencies:
   ```
   pip install langchain langchain-community langchain-ollama chromadb streamlit pypdf python-docx beautifulsoup4
   ```

3. Place documents in the `data/` folder.

4. Run ingestion:
   ```
   python ingestion.py
   ```

5. Run the app:
   ```
   streamlit run app.py
   ```

## Usage
- Enter a question in the text box.
- Click "Get Answer" to retrieve and generate a response.
- View the answer and sources with snippets.

## Demo
The `../streamlit_demo/steamlitLangChain.py` is a separate demo for general RAG.