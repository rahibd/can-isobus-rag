# CAN-RAG: Local RAG + Fine-Tuning for ISOBUS/CAN Documentation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/inference-100%25%20local-orange)

**At a glance:** two AI approaches (RAG vs. QLoRA fine-tuning) built and
compared on the same 200+ document technical corpus, running fully
offline — no cloud APIs, no data leaving the machine. Built by a PhD
researcher in agricultural mechatronics applying the same domain expertise
behind [published ISOBUS communication research](#related-publications)
to make that work's underlying documentation queryable.

A fully offline system for querying dense technical ISOBUS (ISO 11783) and
CAN bus documentation in natural language — built two ways, on purpose, to
compare them: **Retrieval-Augmented Generation (RAG)** against a local
vector store, and **QLoRA fine-tuning** on the same document set, so the
model's own weights learn the domain instead of relying on retrieval at
question time.

Everything runs locally through [LM Studio](https://lmstudio.ai) — no API
keys, no cloud inference, no proprietary documents leaving the machine.

## Table of Contents

- [Why this project](#why-this-project)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Key design decisions](#key-design-decisions)
- [RAG vs. fine-tuning: what I found](#rag-vs-fine-tuning-what-i-found)
- [Engineering challenges](#engineering-challenges)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Related publications](#related-publications)
- [About the developer](#about-the-developer)

## Why this project

Technical specs like SAE J1939 and ISO 11783 are dense, cross-referenced,
and painful to search by keyword. As someone working on ISOBUS-compliant
communication systems for agricultural machinery, I wanted a way to ask
plain-language questions — *"what PGN carries engine speed?"*, *"how does
ECU address claiming work?"* — and get answers grounded in the actual spec
text, without sending proprietary engineering documents to a cloud API.

That became the RAG half of this project. The fine-tuning half exists to
answer a follow-up question directly: **when does it make more sense to
retrieve knowledge at question time versus bake it into the model itself?**

## Screenshots

> *Add 1-2 screenshots or a short GIF of the chat UI in action here —
> e.g. a real question being answered, and the "Not found in the provided
> documents" refusal on an out-of-scope question (shows the grounding
> actually works, not just that it looks nice).*
>
> `![Chat UI](docs/screenshot-chat.png)`

## Architecture

```
                     ┌─────────────────────┐
   .txt / .pdf / .docx │  Chunking + Extraction │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Embedding (CPU)    │  all-MiniLM-L6-v2
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Chroma Vector DB   │
                     └──────┬───────┬───────┘
                            │       │
              ┌─────────────┘       └─────────────┐
              ▼                                    ▼
   ┌─────────────────────┐            ┌─────────────────────────┐
   │   RAG (retrieval)    │            │  Fine-tuning (weights)   │
   │  HyDE-style expansion │            │  auto-generate QA pairs  │
   │  + hybrid retrieval → │            │  → QLoRA on Qwen2.5 /    │
   │  local LLM answer     │            │  Llama-3.1 (Unsloth) →   │
   │  (LM Studio)          │            │  GGUF → back into LM     │
   │                       │            │  Studio                  │
   └─────────────────────┘            └─────────────────────────┘
```

## Features

**RAG pipeline**
- Multi-format ingestion (`.txt`, `.pdf`, `.docx`)
- Local vector search (ChromaDB + sentence-transformers), no cloud embedding API
- Hybrid retrieval — HyDE-style query expansion (hypothetical document
  embeddings: rewriting a terse question into fuller, document-style
  phrasing before embedding it) merged with raw-question search, to hedge
  against a single retrieval pass missing the right chunk
- Fully offline generation via a local LLM served through LM Studio
- Browser chat UI (Flask) and terminal CLI, sharing one core pipeline
- Rebuild-vs-reuse prompt on startup — skip re-embedding when nothing's changed

**Fine-tuning experiment**
- Auto-generates an instruction-tuning dataset (QA pairs) from the same
  document chunks, using the local LLM itself
- QLoRA training on a free Colab T4 GPU via
  [Unsloth](https://github.com/unslothai/unsloth), targeting a full 7-8B model
- Exports to GGUF for direct reuse inside the same LM Studio setup as the RAG side

## Tech stack

Python · Flask · ChromaDB · sentence-transformers · pypdf · python-docx ·
LM Studio · Unsloth (Colab-side fine-tuning)

## Setup

### RAG pipeline

1. Install [LM Studio](https://lmstudio.ai), load a local model (e.g.
   `gemma4:e4b`), and start its Local Server (`http://localhost:1234`).
2. `pip install -r requirements.txt`
3. Drop your own `.txt` / `.pdf` / `.docx` files into `sample_docs/`.
4. Run either interface:
   - Terminal: `python main.py`
   - Web UI: `python app.py`, then open `http://localhost:5000`

On first run the vector store is built from scratch. On later runs you'll
be asked whether to rebuild — answer "N" to reuse the existing index unless
you've changed the source documents.

### Fine-tuning experiment

1. `python dataset_prep.py` — generates `finetune_data.jsonl` from your
   documents (calls the local LM Studio model to write QA pairs; slow on a
   large document set). Uses the same dependencies as the RAG pipeline —
   no extra install needed.
2. Upload `colab_finetune.ipynb` to [Google Colab](https://colab.research.google.com),
   set `Runtime → Change runtime type → T4 GPU`, and run all cells. Trains
   a full 7-8B model via [Unsloth](https://github.com/unslothai/unsloth)
   and exports a GGUF ready to import back into LM Studio.

## Usage

Example questions to try against the RAG chat UI:

- *"What PGN carries engine speed?"*
- *"How does ECU address claiming work?"*
- *"What's the recommended safe bus load threshold?"*
- *"What is a DDOP and what does it describe?"*

## Key design decisions

A few choices made deliberately, and why:

- **Local-first, not just "local because it's free."** Engineering
  documentation is often proprietary. Running inference entirely through
  LM Studio instead of a cloud API means the actual content of the
  documents never leaves the machine — a real constraint in industrial
  contexts, not just a cost optimization.
- **Hybrid retrieval over a single query-expansion pass.** This isn't
  naive, single-pass RAG. Expanding a short question into fuller,
  spec-style phrasing before embedding it — a HyDE-style approach
  ([Gao et al., 2022](https://arxiv.org/abs/2212.10496), *Precise
  Zero-Shot Dense Retrieval without Relevance Labels*) — generally
  improves match quality over embedding the raw question directly. But on
  a small local model, that expansion step has enough variance that the
  *same* question could retrieve different chunks on different runs.
  Merging expanded-query results with raw-query results (deduplicated)
  hedges against either pass drifting off-topic, at the cost of a
  slightly larger context window per answer.
- **RAG and fine-tuning built side by side, not sequentially.** The
  interesting engineering question isn't "can you build a RAG pipeline"
  — it's knowing which approach fits which situation: RAG for
  fast-changing documents where you need traceability to a source, versus
  fine-tuning for stable domain knowledge where per-query latency or
  offline model size matters more than citing exact provenance.
- **Small-model-aware prompting.** Several prompts in this project (JSON
  output formatting, strict grounding instructions) are more explicit and
  repetitive than they'd need to be with a frontier model, because
  smaller local models follow instructions less reliably — that gap had
  to be compensated for in the prompt itself, not assumed away.

## RAG vs. fine-tuning: what I found

> *Fill in once your fine-tuning run completes — this is the section that
> makes the comparison real instead of theoretical. A simple table works
> well:*

| Question | RAG answer | Fine-tuned model answer (no retrieved context) |
|---|---|---|
| What PGN carries engine speed? | ... | ... |
| How does address claiming work? | ... | ... |

## Engineering challenges

A few real issues hit and fixed along the way, kept here because the
debugging is arguably more instructive than the happy path:

- **Corrupt/empty source files** crashing the whole ingestion run —
  isolated with per-file try/except so one bad PDF doesn't take down a
  215-document batch.
- **Chroma's per-call insert limit** (~5,461) — hit at 50,000+ chunks;
  fixed by batching `collection.add()` calls.
- **Non-deterministic retrieval** — query expansion ran at `temperature=0.3`,
  so the *same* question could retrieve different chunks on different runs.
  Fixed with `temperature=0` plus hybrid retrieval (expanded + raw query,
  merged) as a hedge against any single pass drifting off-topic.
- **Small local models don't reliably follow "respond with only JSON"** —
  the fine-tuning dataset generator initially parsed 0 QA pairs from 400
  chunks because the model wrapped output in explanatory prose. Fixed by
  extracting the `[...]` substring directly instead of relying on prefix-based
  stripping.

## Project structure

```
├── main.py                  # terminal RAG chat interface
├── app.py                   # Flask web UI
├── rag_core.py               # shared RAG pipeline (chunking, embedding, retrieval, generation)
├── static/
│   └── index.html           # browser chat UI
├── dataset_prep.py           # generates fine-tuning QA pairs from documents
├── colab_finetune.ipynb      # Colab QLoRA fine-tuning (7-8B model via Unsloth)
├── sample_docs/               # your documents go here (not tracked in git)
└── requirements.txt
```

## Roadmap

- [ ] Telemetry RAG — query CAN-bus log data directly, not just static specs
- [ ] Source citations — surface which document/page an answer came from
- [ ] Sentence-aware chunking instead of raw character counts
- [ ] Complete the RAG-vs-fine-tuning comparison writeup above

## Related publications

This project applies AI tooling to the same domain covered in my published
ISOBUS/CAN research:

- Motalab, M. B., Al-Mallahi, A., Martynenko, A., Al-Tamimi, K., & Paraforos,
  D. S. (2025). *Development of an ISOBUS-compliant communication node for
  multiple machine vision systems on wide boom sprayers for nozzle control
  in spot application schemes.* Smart Agricultural Technology, 100815.
  https://doi.org/10.1016/j.atech.2025.100815
- Motalab, M. B., & Al-Mallahi, A. (2024). *Development of a flexible
  electronic control unit for seamless integration of machine vision to
  CAN-enabled boom sprayers for spot application technology.* Smart
  Agricultural Technology, 100618. https://doi.org/10.1016/j.atech.2024.100618
- Motalab, M. B. (2025). *Development of an ISOBUS-compliant communication
  node for multiple machine vision systems on wide boom sprayers for nozzle
  control in spot application schemes* (PhD dissertation). Dalhousie
  University. https://dalspace.library.dal.ca/items/6b7b0b3f-999e-4bed-865d-2110671c0e92

## About the developer

**Mozammel Bin Motalab, PhD** — Digital Agriculture / Mechatronic Control
Systems researcher, working at the intersection of embedded systems, CAN/ISOBUS
communication, machine vision, and (now) applied AI.

- GitHub: [github.com/rahibd](https://github.com/rahibd)
- LinkedIn: [linkedin.com/in/mbmbd](https://www.linkedin.com/in/mbmbd/)
- Blog: [mmotalab.medium.com](https://mmotalab.medium.com/)

## License

MIT — see `LICENSE`.
