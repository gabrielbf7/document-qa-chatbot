from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = "knowledge_base"
INDEX_DIR = "index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters

TOP_K = 4
