"""
Fine-Tuning Dataset Prep - CAN/ISOBUS
========================================
Turns your existing document chunks into instruction-tuning QA pairs, using
your local LM Studio model to generate the questions and answers. Reuses
the exact same chunking/extraction logic as the RAG project (rag_core.py)
-- same documents, different use: instead of retrieving chunks at question
time, we're "baking" their content into model weights via fine-tuning.

Setup:
  Same as the RAG project -- LM Studio's Local Server must be running.

Output: finetune_data.jsonl, one {"question": ..., "answer": ...} per line.
"""

import json
import random
import requests
from rag_core import discover_files, load_and_chunk_docs, DOCS_DIR, LMSTUDIO_URL, LMSTUDIO_MODEL

SAMPLE_SIZE = 400     # ~500-1000 examples is a typical small fine-tuning set;
                      # some chunks yield 2 pairs, some yield 0, so oversample a bit
OUTPUT_FILE = "finetune_data.jsonl"
MIN_CHUNK_LEN = 200   # skip near-empty/junk chunks (e.g. bad PDF extraction)
DEBUG_SAMPLES = 3     # print the model's raw response for this many chunks,
                      # so you can SEE what shape it's actually returning
                      # instead of guessing when parsing fails


def extract_json_array(raw: str) -> str | None:
    """Pull out just the [...] portion of the response, ignoring any prose
    the model added before/after it (e.g. 'Sure, here are two pairs:').
    Small local models rarely follow 'respond with ONLY JSON' literally."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return raw[start:end + 1]


def generate_qa_pairs(chunk_text: str, debug: bool = False) -> list[dict]:
    """Ask the local model to write 1-2 QA pairs grounded in this chunk."""
    prompt = f"""Read the technical text below and write 1-2 factual question-answer
pairs that test understanding of specific facts it contains (numbers, PGNs,
terminology, procedures). Keep answers concise (1-3 sentences) and grounded
only in the text.

Respond with ONLY the JSON list below -- no explanation, no preamble, no
"here is the answer", nothing before or after it. Your entire reply must
start with the character '[' :

[{{"question": "...", "answer": "..."}}]

Text:
{chunk_text}"""

    response = requests.post(
        LMSTUDIO_URL,
        json={
            "model": LMSTUDIO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 700,   # was 400 -- likely too low if the model adds preamble
                                 # before the JSON and gets cut off before reaching it
        },
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()

    if debug:
        print(f"\n[DEBUG] Raw response ({len(raw)} chars):\n{raw!r}\n{'-' * 40}")

    json_str = extract_json_array(raw)
    if json_str is None:
        if debug:
            print("[DEBUG] No '[...]' found in response at all.")
        return []

    try:
        pairs = json.loads(json_str)
        return [p for p in pairs if isinstance(p, dict) and "question" in p and "answer" in p]
    except (json.JSONDecodeError, TypeError) as e:
        if debug:
            print(f"[DEBUG] Found '[...]' but it didn't parse as JSON: {e}")
        return []


if __name__ == "__main__":
    filepaths = discover_files(DOCS_DIR)
    if not filepaths:
        raise FileNotFoundError(f"No supported files found in {DOCS_DIR}/")

    chunks = load_and_chunk_docs(filepaths)
    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_LEN]
    print(f"{len(chunks)} usable chunk(s) found.")

    sample = random.sample(chunks, min(SAMPLE_SIZE, len(chunks)))
    print(f"Generating QA pairs from {len(sample)} sampled chunk(s) -- this calls "
          f"the local model once per chunk, so it will take a while.\n")

    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(sample, start=1):
            pairs = generate_qa_pairs(chunk["text"], debug=(i <= DEBUG_SAMPLES))
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
                written += 1
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} chunks processed, {written} QA pairs so far")

    print(f"\nDone. Wrote {written} QA pairs to {OUTPUT_FILE}")