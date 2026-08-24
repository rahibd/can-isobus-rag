"""
CAN/ISOBUS Document RAG - Core Pipeline (shared by main.py and app.py)
========================================================================
All the actual RAG logic lives here, with no interactive input() calls,
so both the terminal version (main.py) and the web UI (app.py) can import
and reuse the exact same pipeline.

Pipeline stages:
  1. CHUNKING     -> split raw text into overlapping chunks
  2. EMBEDDING    -> convert each chunk into a vector (runs on CPU, fast)
  3. VECTOR STORE -> store vectors in Chroma, search by similarity
  4. RETRIEVE+GEN -> expand the question, retrieve chunks, generate an answer

Hardware notes (Dell Latitude 5501, i7-9850H, 32GB RAM, MX150 2GB VRAM):
  - Embeddings use 'all-MiniLM-L6-v2' (~80MB, CPU-friendly). No GPU needed.
  - Chroma vector store runs entirely on CPU/disk.
  - Generation uses gemma4:e4b running locally in LM Studio -- no internet,
    no API key, nothing leaves your machine.
"""

import os
import glob
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document

DOCS_DIR = "sample_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5                        # was 3 -- more retrieved context = more material for a full answer
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "gemma4:e4b"    # must match the model name loaded in LM Studio
CHROMA_ADD_BATCH_SIZE = 5000     # Chroma's Rust backend caps a single add() call
MAX_ANSWER_TOKENS = 1024         # was unset -> LM Studio's default cap was truncating answers


# -----------------------------
# 1. CHUNKING + EXTRACTION
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text(filepath: str) -> str:
    """Pull raw text out of a file, regardless of format."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {ext}")


def discover_files(docs_dir: str) -> list[str]:
    """Find every supported file in docs_dir."""
    supported_extensions = ("*.txt", "*.pdf", "*.docx")
    files = []
    for pattern in supported_extensions:
        files.extend(glob.glob(os.path.join(docs_dir, pattern)))
    return sorted(files)


def load_and_chunk_docs(filepaths: list[str]) -> list[dict]:
    """Extract text from each given filepath and chunk it. A single broken
    file is skipped with a warning instead of crashing the whole pipeline."""
    all_chunks = []
    for filepath in filepaths:
        try:
            text = extract_text(filepath)
        except Exception as e:
            print(f"  [skipped] {os.path.basename(filepath)}: {e}")
            continue

        if not text.strip():
            print(f"  [skipped] {os.path.basename(filepath)}: no extractable text")
            continue

        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            all_chunks.append({
                "id": f"{os.path.basename(filepath)}_{i}",
                "text": chunk,
                "source": os.path.basename(filepath),
            })
    return all_chunks


# -----------------------------
# 2 & 3. EMBEDDING + VECTOR STORE
# -----------------------------
def build_vector_store(chunks: list[dict]):
    """Embed all chunks and store them in a local Chroma collection."""
    print("Loading embedding model (CPU)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("can_isobus_docs")
    except Exception:
        pass
    collection = client.create_collection("can_isobus_docs")

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    print(f"Adding to vector store in batches of {CHROMA_ADD_BATCH_SIZE}...")
    for start in range(0, len(chunks), CHROMA_ADD_BATCH_SIZE):
        end = start + CHROMA_ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    return embedder, collection


# -----------------------------
# 4. RETRIEVE + GENERATE
# -----------------------------
def retrieve(question: str, embedder, collection, top_k: int = TOP_K) -> list[str]:
    """Embed the question and pull the most similar chunks from the store."""
    query_vector = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)
    return results["documents"][0]


def merge_chunks(primary: list[str], secondary: list[str], limit: int) -> list[str]:
    """Combine two retrieval result sets, deduplicated, primary first. Used
    to hedge against a single retrieval pass missing the right chunk --
    e.g. an expanded-query search that drifted off-topic still gets backed
    up by a plain search on the raw question."""
    seen = set()
    merged = []
    for chunk in primary + secondary:
        if chunk not in seen:
            seen.add(chunk)
            merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


def expand_query(question: str) -> str:
    """LLM pass #1: rewrite the terse user question into fuller, spec-style
    phrasing, used only for vector search (never shown to the user)."""
    prompt = f"""You are helping search technical documents about CAN bus and ISOBUS.
Write a short (2-3 sentence) hypothetical answer to the question below, using
the kind of technical terms and phrasing that would appear in a spec document.
This text is only used to improve document search -- it will not be shown to
the user, so it's fine if it's not fully accurate.

Question: {question}

Hypothetical answer:"""

    response = requests.post(
        LMSTUDIO_URL,
        json={
            "model": LMSTUDIO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,       # deterministic -- same question should always expand the same way
            "max_tokens": 300,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """LLM pass #2: send retrieved context + the ORIGINAL question to the
    model loaded in LM Studio, and get a full, detailed final answer."""
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are answering questions using ONLY the context below.
Do not use any outside knowledge. If the answer is not in the context,
respond exactly: "Not found in the provided documents."

Give a thorough, detailed answer -- don't just give a one-line reply if the
context supports more detail. Include specific numbers, PGNs, parameter
names, or other technical details from the context where relevant.

Context:
{context}

Question: {question}

Detailed answer (using only the context above):"""

    response = requests.post(
        LMSTUDIO_URL,
        json={
            "model": LMSTUDIO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": MAX_ANSWER_TOKENS,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def build_store_from_docs_dir(docs_dir: str = DOCS_DIR):
    """Convenience wrapper: discover -> chunk -> embed -> store, in one call.
    Always rebuilds from scratch (deletes and recreates the collection)."""
    filepaths = discover_files(docs_dir)
    if not filepaths:
        raise FileNotFoundError(f"No supported files found in {docs_dir}/")
    print(f"Found {len(filepaths)} file(s) in {docs_dir}/:")
    for f in filepaths:
        print(f"  - {os.path.basename(f)}")
    chunks = load_and_chunk_docs(filepaths)
    print(f"Loaded and chunked into {len(chunks)} pieces from {len(filepaths)} file(s)")
    return build_vector_store(chunks)


def load_existing_collection():
    """Try to load a previously-built vector store from disk WITHOUT
    re-embedding anything. Returns (embedder, collection), or None if no
    usable existing store is found."""
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection("can_isobus_docs")
    except Exception:
        return None

    if collection.count() == 0:
        return None

    print("Loading embedding model (CPU)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder, collection


def get_vector_store(force_rebuild: bool = False, docs_dir: str = DOCS_DIR):
    """Load the existing vector store from disk if one exists and a
    rebuild wasn't requested. Otherwise (or if none exists yet) does the
    full discover -> chunk -> embed -> store pipeline."""
    if not force_rebuild:
        existing = load_existing_collection()
        if existing is not None:
            embedder, collection = existing
            print(f"Loaded existing vector store with {collection.count()} chunk(s) -- skipped re-embedding.")
            return embedder, collection
        print("No existing vector store found -- building from scratch.")

    return build_store_from_docs_dir(docs_dir)
