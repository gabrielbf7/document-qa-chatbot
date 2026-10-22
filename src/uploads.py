import re
from pathlib import Path

from src.config import KNOWLEDGE_DIR, ROOT_DIR, UPLOADS_DIR
from src.loaders import SUPPORTED_EXTENSIONS


def sanitize_filename(name: str) -> str | None:
    """Keep only the base name with safe characters; None if nothing usable remains."""
    base = name.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if not base or Path(base).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    return base


def save_upload(name: str, data: bytes) -> Path | None:
    """Save an uploaded file into uploads/; returns the saved path or None if rejected."""
    safe_name = sanitize_filename(name)
    if safe_name is None:
        return None
    # Source names must stay unique across knowledge_base/ and uploads/.
    if (ROOT_DIR / KNOWLEDGE_DIR / safe_name).exists():
        safe_name = f"upload_{safe_name}"
    uploads_dir = ROOT_DIR / UPLOADS_DIR
    uploads_dir.mkdir(exist_ok=True)
    path = uploads_dir / safe_name
    path.write_bytes(data)
    return path
