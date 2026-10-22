import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
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


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk loaded documents; adds chunk_id, chunk_index, source, page, section."""
    chunks, counters = [], {}
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            source = doc["source"]
            index = counters.get(source, 0)
            counters[source] = index + 1
            chunks.append(
                {
                    "chunk_id": f"{source}#{index}",
                    "source": source,
                    "page": doc["page"],
                    "section": chunk["section"],
                    "chunk_index": index,
                    "text": chunk["text"],
                }
            )
    return chunks
