Offline RAG Assistant

Private, offline Retrieval-Augmented Generation using Ollama + FAISS + Python, with a Streamlit UI(to make it available online as well if needed), automatic document ingestion, and local LLM inference.

This project demonstrates how to build a privacy-first AI assistant that performs semantic search, retrieval, and grounded generation without ever sending data to the cloud.

⭐ Features
🔐 100% Offline

All embeddings generated locally using Ollama (nomic-embed-text)

All generation done with Llama 3.2 (2GB local)

No external APIs

No internet required after model download

📄 Multi-format Document Support

PDF

Markdown

HTML

TXT

DOCX

🔎 Smart Retrieval

Overlapping chunking

Cosine similarity search

FAISS vector index

Dynamic thresholding

Parallel text cleaning

🧠 Grounded Generation

Context-aware answers

Source citations

Guaranteed “don’t hallucinate” rules

🖥️ Modern Streamlit UI

Chat interface

File uploader

Auto-reindex when new docs are uploaded

Retrieval visualization

Confidence scoring
