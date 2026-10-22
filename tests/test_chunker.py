from src.chunker import chunk_documents, is_caps_heading, split_sections
from src.config import CHUNK_SIZE

MD = """# Our Team

## Dentists

### Dr. Park — General
Exams and cleanings.

### Dr. Reed — Endodontics
Root canals.

## Hygienists
- Jordan, cleanings.
"""


def md_doc(text=MD, source="team.md"):
    return [{"source": source, "page": None, "text": text}]


def test_markdown_splits_at_every_heading_with_body():
    chunks = chunk_documents(md_doc())
    assert [c["section"] for c in chunks] == ["Dr. Park — General", "Dr. Reed — Endodontics", "Hygienists"]
    assert "Root canals." in chunks[1]["text"] and "Exams" not in chunks[1]["text"]


def test_heading_only_sections_are_dropped_but_kept_in_path():
    sections = split_sections(MD, markdown=True)
    assert [s["titles"][-1] for s in sections] == ["Dr. Park — General", "Dr. Reed — Endodontics", "Hygienists"]
    assert sections[0]["titles"] == ["Our Team", "Dentists", "Dr. Park — General"]


def test_header_is_title_and_section_path():
    chunks = chunk_documents(md_doc())
    assert chunks[0]["header"] == "Our Team > Dentists > Dr. Park — General"
    assert chunks[2]["header"] == "Our Team > Hygienists"


def test_title_falls_back_to_file_name_without_h1():
    chunks = chunk_documents(md_doc("## Hours\nOpen daily.", source="hours_and_location.md"))
    assert chunks[0]["header"] == "Hours And Location > Hours"


def test_overlap_never_crosses_section_boundary():
    paragraphs = [
        " ".join(f"Alpha {p}.{i} sentence ends here." for i in range(9)) for p in range(4)
    ]  # ~300 chars each: merged two per chunk, so overlap kicks in
    long_a = "\n\n".join(paragraphs)
    text = f"# Doc\n\n## A\n{long_a}\n\n## B\nBeta only text."
    chunks = chunk_documents(md_doc(text))
    a_chunks = [c for c in chunks if c["section"] == "A"]
    b_chunks = [c for c in chunks if c["section"] == "B"]
    assert len(a_chunks) > 1  # long section is split, with overlap inside A
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)
    assert len(b_chunks) == 1 and "Alpha" not in b_chunks[0]["text"]
    assert all("Beta" not in c["text"] for c in a_chunks)
    assert a_chunks[1]["text"][:40] in a_chunks[0]["text"]  # overlap within the section


def test_txt_caps_headings_split_sections():
    text = "CLINIC - HOURS\n\nIntro line.\n\nOPENING HOURS\nMonday: 8 AM\nSunday: Closed\n\nPARKING\nFree garage."
    chunks = chunk_documents([{"source": "hours.txt", "page": None, "text": text}])
    assert [c["section"] for c in chunks] == ["Clinic - Hours", "Opening Hours", "Parking"]
    assert chunks[1]["header"] == "Clinic - Hours > Opening Hours"
    assert "Sunday: Closed" in chunks[1]["text"] and "Free garage" not in chunks[1]["text"]


def test_txt_non_heading_lines_are_not_split():
    assert is_caps_heading("OPENING HOURS")
    assert is_caps_heading("1. GREETING")
    assert not is_caps_heading("Phone: (541) 555-0148")
    assert not is_caps_heading("- NOT A HEADING")
    assert not is_caps_heading("911")
    assert not is_caps_heading("")


def test_pdf_header_has_title_and_page_and_no_section():
    docs = [{"source": "privacy_policy.pdf", "page": 2, "text": "Some policy text."}]
    chunk = chunk_documents(docs)[0]
    assert chunk["header"] == "Privacy Policy > p. 2"
    assert chunk["section"] is None and chunk["page"] == 2


def test_chunk_ids_are_sequential_per_source():
    chunks = chunk_documents(md_doc())
    assert [c["chunk_id"] for c in chunks] == ["team.md#0", "team.md#1", "team.md#2"]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
