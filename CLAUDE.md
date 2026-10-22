# Document Q&A Chatbot

Portfolio project: a RAG chatbot (Python, Claude API, Streamlit) that answers questions about the documents of a fictional dental clinic, Maplewood Dental Studio.

## MVP scope

- Load the documents in `knowledge_base/` (PDF, MD, TXT).
- Split into chunks, build a searchable index, retrieve relevant chunks for each question.
- Answer with the Claude API using only the retrieved context, citing the source document.
- Simple Streamlit chat interface.
- Out of scope: authentication, databases, multi-user support, real patient data.

## Folder structure

```
knowledge_base/   fictional source documents (4 PDF, 3 MD, 3 TXT) - do not edit
index/            generated search index (git-ignored)
uploads/          user uploads (git-ignored)
.venv/            virtual environment (git-ignored)
.claude/          Claude Code settings
.env.example      template for the API key
```

## Rules

- Never read `.env` and never display API keys.
- Never run git commands; the user runs them.
- Use only fictional documents.
- Keep to the MVP scope.
- Explain in Portuguese before running any command.

## Architecture

- `src/config.py`: all constants (paths, model, chunk size/overlap, TOP_K).
- `src/loaders.py`: MD/TXT as one document, PDFs page by page (page number kept). The repeated PDF header (clinic address/phone line and "Page N") is stripped by regex.
- `src/chunker.py`: split by paragraph, merge up to 800 chars, ~150 chars of overlap, oversized paragraphs split by sentence. The latest Markdown heading is stored as `section`.
- `src/indexer.py`: embeddings with `intfloat/multilingual-e5-small` (multilingual, small, runs locally on CPU). Passages use the `passage: ` prefix and queries use `query: `. Vectors are normalized, so cosine similarity is a plain dot product (numpy, no vector DB).
- Index files in `index/` (git-ignored): `embeddings.npy`, `chunks.json` (chunk_id like `faq.md#2`, source, page, section, chunk_index, text) and `manifest.json` (model, chunk settings, created_at, sha256 per file).
- CLI: `python -m src.indexer build` and `python -m src.indexer search "question"`.
- Known limit: e5-small scores are compressed (about 0.77-0.84), so a fixed score threshold is unreliable for detecting "no answer". Let the LLM decide from the retrieved context.
