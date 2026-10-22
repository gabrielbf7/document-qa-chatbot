from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = "knowledge_base"
INDEX_DIR = "index"
UPLOADS_DIR = "uploads"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters

TOP_K = 8

LLM_MODEL = "claude-haiku-5-5"
LLM_EFFORT = "low"
MAX_OUTPUT_TOKENS = 1024

# USD per million tokens
PRICE_INPUT_PER_MTOK = 0.10
PRICE_OUTPUT_PER_MTOK = 0.50
