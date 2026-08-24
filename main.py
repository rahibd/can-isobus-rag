"""
CAN/ISOBUS Document RAG - Terminal Version
============================================
Same pipeline as app.py (the web UI), just running as a plain terminal
loop instead of a browser chat window. All the real logic lives in
rag_core.py -- this file just wires it up and handles input().

Setup:
  1. In LM Studio: load gemma4:e4b, then start the Local Server
     (default: http://localhost:1234)
  2. pip install -r requirements.txt
  3. python main.py
"""

from rag_core import get_vector_store, expand_query, retrieve, merge_chunks, generate_answer

if __name__ == "__main__":
    choice = input("Rebuild the vector store from sample_docs/? Only needed if files "
                    "were added/changed. [y/N]: ").strip().lower()
    embedder, collection = get_vector_store(force_rebuild=(choice == "y"))

    print("\nRAG pipeline ready. Type a question (or 'quit').\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit"):
            break

        search_text = expand_query(question)
        print(f"\n[Expanded search text: {search_text[:120]}...]")

        expanded_chunks = retrieve(search_text, embedder, collection)
        raw_chunks = retrieve(question, embedder, collection)
        top_chunks = merge_chunks(expanded_chunks, raw_chunks, limit=8)
        print(f"[Retrieved {len(top_chunks)} relevant chunk(s)]\n")

        answer = generate_answer(question, top_chunks)
        print(f"Answer:\n{answer}\n")
        print("-" * 60)
