import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# Every PDF page starts with the clinic address/phone line followed by "Page N".
PDF_FOOTER_LINE = re.compile(r"^Maplewood Dental Studio\b.*\(\d{3}\) \d{3}-\d{4}\s*$")
PDF_PAGE_LINE = re.compile(r"^Page \d+\s*$")


def clean_pdf_page(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not PDF_FOOTER_LINE.match(line.strip()) and not PDF_PAGE_LINE.match(line.strip())
    ]
    return "\n".join(lines).strip()


def load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_pdf_page(page.extract_text() or "")
        if text:
            pages.append({"source": path.name, "page": number, "text": text})
    return pages


def load_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    return [{"source": path.name, "page": None, "text": text}] if text else []


def load_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text(path)


def load_documents(directory: Path) -> list[dict]:
    """Return one entry per page (PDF) or per file (MD/TXT)."""
    documents = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))
    return documents
