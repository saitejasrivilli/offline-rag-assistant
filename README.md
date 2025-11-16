Offline RAG Assistant

A private, fully offline Retrieval-Augmented Generation system built with Ollama, FAISS, and Python, featuring a Streamlit UI, intelligent chunking, and complete local inference.
This project proves that powerful AI assistants can run entirely on your own machine without sending any data to the cloud.

⭐ Features
🔐 Completely Offline

Local embeddings with nomic-embed-text

Local LLM generation with Llama 3.2

No API keys

No external network calls

Your data never leaves your device

📄 Multi-Format Document Support

Supports:

PDF

Markdown

HTML

TXT

DOCX (optional add-on)

🔎 Retrieval Components

Chunking with overlap

Sentence-aware boundaries

Cosine similarity search

FAISS vector index

Adjustable thresholds

Transparent chunk metadata

🧠 Answer Generation

Llama 3.2 for grounded responses

Strict “no hallucination” prompt rules

Cited answers with source tracking

Confidence scoring

🖥️ Streamlit UI

Chat interface

File uploader

Automatic re-indexing

Retrieval visualization

Clean, minimal layout

🚀 Quick Start
1. Clone the repository
git clone https://github.com/saitejasrivilli/offline-rag-assistant.git
cd offline-rag-assistant

2. Install dependencies
pip install -r requirements.txt

3. Install Ollama

Download from: https://ollama.com/download

Verify:

ollama --version

4. Pull the models
ollama pull llama3.2
ollama pull nomic-embed-text

5. Run the core RAG engine
python3 rag.py

6. Run the UI
streamlit run app.py

🧩 Project Structure
offline-rag-assistant/
│── app.py                # Streamlit UI
│── rag.py                # Core offline RAG engine
│── requirements.txt
│── documents/            # User documents stored here
│── vector_db/            # FAISS index + metadata
│── README.md
└── .gitignore
