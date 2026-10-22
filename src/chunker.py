import re
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TXT_HEADING_MAX_LEN = 60
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, max_len: int) -> list[str]:
    """Split an oversized paragraph into pieces of at most max_len characters."""
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(text):
        # A single sentence longer than max_len is cut by words.
        while len(sentence) > max_len:
            cut = sentence.rfind(" ", 0, max_len)
            cut = cut if cut > 0 else max_len
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_len:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def to_units(text: str) -> list[tuple[str, str | None]]:
    """Split text into (paragraph, section) units, tracking the latest Markdown heading."""
    units, section = [], None
    for block in re.split(r"\n\s*\n", text):
        # Break a block where a heading starts mid-block, so the section stays accurate.
        parts, current = [], []
        for line in block.splitlines():
            if HEADING.match(line) and current:
                parts.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            parts.append("\n".join(current))

        for part in parts:
            part = part.strip()
            if not part:
                continue
            first_line = part.splitlines()[0]
            match = HEADING.match(first_line)
            if match:
                section = match.group(1)
            if len(part) > CHUNK_SIZE:
                units.extend((piece, section) for piece in split_sentences(part, CHUNK_SIZE))
            else:
                units.append((part, section))
    return units


def overlap_tail(text: str) -> str:
    """Last ~CHUNK_OVERLAP characters of text, starting at a word boundary."""
    if len(text) <= CHUNK_OVERLAP:
        return text
    tail = text[-CHUNK_OVERLAP:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_text(text: str) -> list[dict]:
    """Return chunks as {"text", "section"}; merges paragraphs up to CHUNK_SIZE."""
    chunks = []
    current, section, has_new = "", None, False

    for unit, unit_section in to_units(text):
        if current and len(current) + 2 + len(unit) > CHUNK_SIZE:
            if has_new:
                chunks.append({"text": current, "section": section})
                current = overlap_tail(current)
            else:
                current = ""  # only overlap text so far: drop it rather than exceed the size
            has_new = False
            if current and len(current) + 2 + len(unit) > CHUNK_SIZE:
                current = ""
        if not has_new:
            section = unit_section
        current = f"{current}\n\n{unit}" if current else unit
        has_new = True

    if current and has_new:
        chunks.append({"text": current, "section": section})
    return chunks


def is_caps_heading(line: str) -> bool:
    """TXT heading: a short line with letters and no lowercase, e.g. "OPENING HOURS" or "1. GREETING"."""
    stripped = line.strip()
    return (
        0 < len(stripped) <= TXT_HEADING_MAX_LEN
        and any(char.isalpha() for char in stripped)
        and stripped == stripped.upper()
        and not stripped.startswith("-")
    )


def split_sections(text: str, markdown: bool) -> list[dict]:
    """Split text at headings into sections: {"titles", "levels", "text"}.

    titles/levels is the heading path (parents first). Sections whose body is empty
    (a heading directly followed by a sub-heading) are dropped; their title still
    appears in the path of their children.
    """
    sections, stack, current, seen_text = [], [], None, False
    for line in text.splitlines():
        level = title = None
        if markdown:
            match = MD_HEADING.match(line)
            if match:
                level, title = len(match.group(1)), match.group(2)
        elif is_caps_heading(line):
            level, title = (2 if seen_text else 1), line.strip().title()
        seen_text = seen_text or bool(line.strip())

        if level:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = {
                "levels": [lvl for lvl, _ in stack],
                "titles": [ttl for _, ttl in stack],
                "lines": [line],
                "body": 0,
            }
            sections.append(current)
        else:
            if current is None:
                current = {"levels": [], "titles": [], "lines": [], "body": 0}
                sections.append(current)
            current["lines"].append(line)
            current["body"] += bool(line.strip())

    return [
        {"titles": s["titles"], "levels": s["levels"], "text": "\n".join(s["lines"]).strip()}
        for s in sections
        if s["body"]
    ]


def humanize(source: str) -> str:
    return Path(source).stem.replace("_", " ").title()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk loaded documents.

    MD/TXT are split at headings first, then each section goes through chunk_text, so
    overlap never crosses a section boundary. Each chunk gets chunk_id, chunk_index,
    source, page, section (innermost heading) and header ("<doc title> > <section path>",
    or "<doc title> > p. N" for PDFs), which is prepended to the text before embedding.
    """
    chunks, counters = [], {}
    for doc in documents:
        source, page = doc["source"], doc["page"]
        suffix = Path(source).suffix.lower()
        if suffix in (".md", ".txt"):
            sections = split_sections(doc["text"], markdown=suffix == ".md")
        else:
            sections = [{"titles": [], "levels": [], "text": doc["text"]}]

        # Document title: the level-1 heading if there is one, else the file name.
        title = next((s["titles"][0] for s in sections if s["levels"][:1] == [1]), humanize(source))

        for section in sections:
            titles = section["titles"]
            path = titles[1:] if section["levels"][:1] == [1] else titles
            if page is not None:
                header = f"{title} > p. {page}"
            else:
                header = " > ".join([title, *path])

            for chunk in chunk_text(section["text"]):
                index = counters.get(source, 0)
                counters[source] = index + 1
                chunks.append(
                    {
                        "chunk_id": f"{source}#{index}",
                        "source": source,
                        "page": page,
                        "section": titles[-1] if titles else None,
                        "chunk_index": index,
                        "header": header,
                        "text": chunk["text"],
                    }
                )
    return chunks
