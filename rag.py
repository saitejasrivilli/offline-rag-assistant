import os
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import PyPDF2
from bs4 import BeautifulSoup
import markdown
import numpy as np
import faiss


# ============================================================
# Chunk Dataclass
# ============================================================

@dataclass
class Chunk:
    id: str
    text: str
    vector: Optional[np.ndarray]
    metadata: Dict


# ============================================================
# Document Loader
# ============================================================

class DocumentLoader:

    @staticmethod
    def load_pdf(file_path: str) -> List[Dict]:
        chunks = []
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "source": os.path.basename(file_path),
                                "page": page_num + 1,
                                "type": "pdf"
                            }
                        })
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return chunks

    @staticmethod
    def load_markdown(file_path: str) -> List[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = markdown.markdown(f.read())
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text()
                return [{
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": 1,
                        "type": "markdown"
                    }
                }]
        except Exception as e:
            print(f"Error loading markdown {file_path}: {e}")
            return []
    @staticmethod
    def load_txt(file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return [{
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "type": "txt"
                }
            }]
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []
    from docx import Document

    @staticmethod
    def load_docx(file_path: str):
        try:
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [{
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "type": "docx"
                }
            }]
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []
    from ebooklib import epub

    @staticmethod
    def load_epub(file_path: str):
        try:
            book = epub.read_epub(file_path)
            text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    text += soup.get_text() + "\n"
            return [{
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": 1,
                    "type": "epub"
                }
            }]
        except Exception as e:
            print(f"Error loading EPUB {file_path}: {e}")
            return []

    @staticmethod
    def load_html(file_path: str) -> List[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text()
                return [{
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": 1,
                        "type": "html"
                    }
                }]
        except Exception as e:
            print(f"Error loading HTML {file_path}: {e}")
            return []

    @staticmethod
    def load_documents(directory: str) -> List[Dict]:
        documents = []
        doc_dir = Path(directory)

        if not doc_dir.exists():
            print(f"Creating {directory}...")
            doc_dir.mkdir(parents=True)
            print("Add documents to the folder and run again.")
            return documents

        for file_path in doc_dir.rglob("*"):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext == ".pdf":
                documents.extend(DocumentLoader.load_pdf(str(file_path)))
            elif ext in [".md", ".markdown"]:
                documents.extend(DocumentLoader.load_markdown(str(file_path)))
            elif ext in [".html", ".htm"]:
                documents.extend(DocumentLoader.load_html(str(file_path)))
            elif ext == '.txt':
                documents.extend(DocumentLoader.load_txt(str(file_path)))
            elif ext == '.docx':
                documents.extend(DocumentLoader.load_docx(str(file_path)))
            elif ext == '.epub':
                documents.extend(DocumentLoader.load_epub(str(file_path)))


        print(f"Loaded {len(documents)} document sections")
        return documents


# ============================================================
# Text Chunker
# ============================================================

class TextChunker:

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\,\!\?\-\:\;]", "", text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 750, overlap: int = 100, metadata: Dict = None) -> List[Chunk]:
        text = TextChunker.clean_text(text)
        if not text:
            return []

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                search_start = end - int(chunk_size * 0.2)
                boundary = max(
                    text.rfind(".", search_start, end),
                    text.rfind("!", search_start, end),
                    text.rfind("?", search_start, end)
                )
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_meta = metadata.copy() if metadata else {}
                chunk_meta["chunk_index"] = idx
                chunk_id = f"{chunk_meta.get('source','unknown')}_{idx}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    vector=None,
                    metadata=chunk_meta
                ))
                idx += 1

            start = end - overlap
            if start >= len(text) - overlap:
                break

        return chunks


# ============================================================
# Embeddings using Ollama
# ============================================================

class OllamaEmbedder:

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self._verify_model()

    def _verify_model(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            if self.model_name not in result.stdout:
                raise RuntimeError(f"Model {self.model_name} not installed. Run: ollama pull {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Ollama is not available: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        try:
            import http.client
            conn = http.client.HTTPConnection("localhost", 11434, timeout=30)
            payload = json.dumps({"model": self.model_name, "prompt": text})
            conn.request("POST", "/api/embeddings", payload, {"Content-Type": "application/json"})
            data = json.loads(conn.getresponse().read())
            return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            return np.zeros(768, dtype=np.float32)

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        print(f"Embedding {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            if i % 10 == 0 and i > 0:
                print(f"Progress: {i}/{len(chunks)}")
            chunk.vector = self.embed_text(chunk.text)
        print("Embeddings complete.")
        return chunks


# ============================================================
# FAISS Vector Database
# ============================================================

class VectorDatabase:

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]):
        vectors = np.array([c.vector for c in chunks], dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} chunks (total: {len(self.chunks)})")

    def search(self, q_vector: np.ndarray, top_k=5):
        q = q_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, top_k)

        results = []
        for i, s in zip(idxs[0], sims[0]):
            if i < len(self.chunks):
                distance = 1 - float(s)
                results.append((self.chunks[i], distance))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, f"{directory}/faiss.index")
        with open(f"{directory}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in self.chunks],
                f,
                indent=2
            )
        print(f"Database saved to {directory}")

    def load(self, directory: str, embedder) -> bool:
        idx_path = f"{directory}/faiss.index"
        meta_path = f"{directory}/chunks.json"

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            print("No saved database found.")
            return False

        self.index = faiss.read_index(idx_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            stored = json.load(f)

        print("Re-embedding chunks...")
        self.chunks = []
        for item in stored:
            vec = embedder.embed_text(item["text"])
            self.chunks.append(Chunk(item["id"], item["text"], vec, item["metadata"]))

        print(f"Loaded {len(self.chunks)} chunks.")
        return True


# ============================================================
# LLM Wrapper
# ============================================================

class OllamaLLM:

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self._verify_model()

    def _verify_model(self):
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            if self.model_name not in out.stdout:
                raise RuntimeError(f"Model {self.model_name} not installed.")
        except Exception as e:
            raise RuntimeError(f"Ollama unavailable: {e}")

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# RAG System
# ============================================================

class RAGSystem:

    def __init__(self, documents_dir="documents", db_dir="vector_db",
                 llm_model="llama3.2", embedding_model="nomic-embed-text"):

        print("Initializing RAG system...")
        self.documents_dir = documents_dir
        self.db_dir = db_dir
        self.embedder = OllamaEmbedder(embedding_model)
        self.llm = OllamaLLM(llm_model)
        self.vector_db = VectorDatabase()
        print("RAG system ready.")

    def ingest_documents(self, chunk_size=750, overlap=100, force_rebuild=False):

        if not force_rebuild:
            if os.path.exists(self.db_dir) and self.vector_db.load(self.db_dir, self.embedder):
                return

        print("Building new vector database...")

        docs = DocumentLoader.load_documents(self.documents_dir)
        if not docs:
            print("No documents found.")
            return

        all_chunks = []
        for d in docs:
            chunks = TextChunker.chunk_text(d["text"], chunk_size, overlap, d["metadata"])
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks")

        all_chunks = self.embedder.embed_chunks(all_chunks)
        self.vector_db.add_chunks(all_chunks)
        self.vector_db.save(self.db_dir)

    def query(self, question: str, top_k=5, distance_threshold=1.5):

        print(f"Question: {question}")

        q_vec = self.embedder.embed_text(question)
        results = self.vector_db.search(q_vec, top_k)

        filtered = [(c, d) for c, d in results if d < distance_threshold]
        if not filtered:
            return {"answer": "Not enough context.", "sources": [], "confidence": "low"}

        context = ""
        sources_info = []

        for i, (chunk, dist) in enumerate(filtered, 1):
            context += f"[Source {i}: {chunk.metadata['source']}, Page {chunk.metadata.get('page','N/A')}]\n"
            context += chunk.text + "\n\n"
            sources_info.append({
                "id": chunk.id,
                "source": chunk.metadata["source"],
                "page": chunk.metadata.get("page", "N/A"),
                "distance": dist
            })

        prompt = f"""
Answer the question strictly from the context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Use only the context.
- Cite sources like "According to Source 1".
- If the context is insufficient, say so.

ANSWER:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources_info,
            "confidence": "high" if len(filtered) >= 3 else "medium"
        }


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    rag = RAGSystem()
    rag.ingest_documents(force_rebuild=True)

    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        result = rag.query(q)
        print("\nANSWER:", result["answer"])

