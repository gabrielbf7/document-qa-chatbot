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
- `src/chunker.py`: MD and TXT are first split into sections at headings (MD: `#` lines; TXT: short ALL-CAPS lines such as "OPENING HOURS" or "1. GREETING", the first one being the doc title). Heading-only sections are dropped (their title stays in the children's path). Inside each section: split by paragraph, merge up to 800 chars, ~150 chars of overlap (never across sections), oversized paragraphs split by sentence (no overlap between those pieces). PDFs are chunked per page, no headings. `section` = innermost heading (None for PDFs).
- Each chunk has a `header` = "<doc title> > <section path>" (PDFs: "<doc title> > p. N"; title = H1/first TXT line, else the humanized file name). The header is prepended to the text only for the embedding (`passage: <header>\n<text>`); the LLM prompt and the UI still use `text` and `section`.
- `src/indexer.py`: embeddings with `intfloat/multilingual-e5-small` (multilingual, small, runs locally on CPU). Passages use the `passage: ` prefix and queries use `query: `. Vectors are normalized, so cosine similarity is a plain dot product (numpy, no vector DB).
- Index files in `index/` (git-ignored): `embeddings.npy`, `chunks.json` (chunk_id like `faq.md#2`, source, page, section, chunk_index, header, text) and `manifest.json` (model, chunk settings, created_at, sha256 per file).
- CLI: `python -m src.indexer build` and `python -m src.indexer search "question"`.
- `src/qa.py`: `answer(question)` runs the search (TOP_K=8), sends the chunks as `<document id source section page>` tags with the question last, and asks Claude (`claude-haiku-5-5`, `output_config.effort="low"`, max 1024 output tokens) for JSON only: `{"answered", "answer", "sources": [chunk_id]}`. Code, not the model, maps chunk ids to source/section/page.
- Answer rules (system prompt): use only the documents, reply in the question's language, say "don't know" and point to the front desk when the answer is missing, never give medication doses or outside knowledge.
- Guardrails in code: chunk ids not retrieved are dropped; malformed JSON, or an "answered" reply citing no valid chunk, becomes `answered=false` with a fallback message; `answered=false` always has no sources.
- Cost: `cost_usd` comes from `response.usage` and the prices in `config.py` (USD per million tokens).
- The API key is loaded from `.env` with python-dotenv; `qa.py` fails with a clear message if it is missing.
- CLI: `python -m src.qa ask "question"` and `python -m src.qa eval` (10 EN/PT questions, real API calls, about $0.002 total).
- `app.py` (Streamlit UI, run with `streamlit run app.py`): chat with `st.chat_input`/`st.chat_message`, history in `st.session_state`. Each answer shows sources as "file — section — p. N" (missing parts omitted), its cost, and the session total in the sidebar. `answered=false` shows the model's "don't know / contact the front desk" reply with no sources.
- UI caching: `st.cache_resource` for the embedding model, the Anthropic client and the loaded index (`answer(..., index=...)` and `search(..., index=...)` take the cached index). The Reindex button rebuilds from `knowledge_base/` + `uploads/` and clears only the index cache. If `index/` is missing, the app shows a warning and a "Build index" button.
- `src/uploads.py`: uploads are saved to `uploads/` with a sanitized base name (only .pdf/.txt/.md, unsafe characters replaced); a name that already exists in `knowledge_base/` gets an `upload_` prefix so sources stay unique. Uploaded files only enter searches after Reindex.
- Tests: `python -m pytest` (no API calls; covers prompt building, cost, JSON parsing).
- TOP_K=8 (raised from 4): about $0.0002 per question with Haiku 5.5, eval 10/10.
- Known miss: "list all" questions may miss items. For "Quais são as especialidades dos dentistas da clínica?" the dentist chunks rank 5th (Dr. Nair), 9th (Dr. Park) and 20th (Dr. Reed), so with TOP_K=8 Dr. Park is missing from the answer and Dr. Reed only appears through the "Choosing a dentist" chunk. Dr. Reed ranks low in PT (12th in EN). Candidates: keyword/hybrid search, or a larger TOP_K for list-style questions.
- Known limit: e5-small scores are compressed (about 0.77-0.84), so a fixed score threshold is unreliable for detecting "no answer". Let the LLM decide from the retrieved context.
