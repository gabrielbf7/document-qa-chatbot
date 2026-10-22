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
