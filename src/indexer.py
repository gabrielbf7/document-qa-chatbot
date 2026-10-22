import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

from src.chunker import chunk_documents
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    KNOWLEDGE_DIR,
    ROOT_DIR,
    TOP_K,
)
from src.loaders import SUPPORTED_EXTENSIONS, load_documents

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str], prefix: str) -> np.ndarray:
    vectors = get_model().encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index() -> list[dict]:
    knowledge_dir = ROOT_DIR / KNOWLEDGE_DIR
    index_dir = ROOT_DIR / INDEX_DIR
    index_dir.mkdir(exist_ok=True)

    chunks = chunk_documents(load_documents(knowledge_dir))
    embeddings = embed([chunk["text"] for chunk in chunks], "passage: ")

    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            path.name: sha256_of(path)
            for path in sorted(knowledge_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        },
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return chunks


def load_index() -> tuple[np.ndarray, list[dict]]:
    index_dir = ROOT_DIR / INDEX_DIR
    embeddings = np.load(index_dir / "embeddings.npy")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return embeddings, chunks


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    embeddings, chunks = load_index()
    query_vector = embed([query], "query: ")[0]
    scores = embeddings @ query_vector  # vectors are normalized, so dot product = cosine
    best = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if argv and argv[0] == "build":
        chunks = build_index()
        print(f"Indexed {len(chunks)} chunks into {INDEX_DIR}/")
        return 0
    if len(argv) >= 2 and argv[0] == "search":
        for result in search(" ".join(argv[1:])):
            preview = result["text"][:150].replace("\n", " ")
            print(f'{result["score"]:.3f}  {result["chunk_id"]}  [{result["section"]}]  {preview}')
        return 0
    print('Usage: python -m src.indexer build | search "question"')
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
