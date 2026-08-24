"""
CAN/ISOBUS Document RAG - Web UI
===================================
A local chat interface for the same RAG pipeline used in main.py.
Builds the vector store once at startup, then serves a browser chat UI
that calls /ask for each question.

Setup:
  1. In LM Studio: load gemma4:e4b, then start the Local Server
     (default: http://localhost:1234)
  2. pip install -r requirements.txt
  3. python app.py
  4. Open http://localhost:5000 in your browser
"""

from flask import Flask, request, jsonify, send_from_directory
from rag_core import get_vector_store, expand_query, retrieve, merge_chunks, generate_answer

app = Flask(__name__, static_folder="static")

# Built once at startup -- rebuilding per-request would re-embed everything
# on every message, which is both slow and pointless since the docs
# haven't changed. Reusing a prior run's store (if one exists) skips the
# slow re-embedding step entirely unless you explicitly ask for a rebuild.
_choice = input("Rebuild the vector store from sample_docs/? Only needed if files "
                 "were added/changed. [y/N]: ").strip().lower()
print("Starting up...")
embedder, collection = get_vector_store(force_rebuild=(_choice == "y"))
print("Ready. Open http://localhost:5000 in your browser.\n")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        search_text = expand_query(question)
        print(f"\n[DEBUG] Question: {question}")
        print(f"[DEBUG] Expanded search text: {search_text[:200]}")

        expanded_chunks = retrieve(search_text, embedder, collection)
        print(f"[DEBUG] Retrieved {len(expanded_chunks)} chunk(s) using EXPANDED text:")
        for i, c in enumerate(expanded_chunks):
            print(f"[DEBUG]   expanded[{i}]: {c[:150]}")

        raw_chunks = retrieve(question, embedder, collection)
        print(f"[DEBUG] Retrieved {len(raw_chunks)} chunk(s) using RAW question:")
        for i, c in enumerate(raw_chunks):
            print(f"[DEBUG]   raw[{i}]: {c[:150]}")

        # Merge both -- hedges against a single retrieval pass missing the
        # right chunk, which is what was causing the flip-flopping answers.
        chunks = merge_chunks(expanded_chunks, raw_chunks, limit=8)
        print(f"[DEBUG] Using {len(chunks)} merged chunk(s) for the final answer")

        answer = generate_answer(question, chunks)
    except Exception as e:
        # Most common cause: LM Studio's Local Server isn't running.
        return jsonify({"error": f"Generation failed: {e}"}), 502

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
