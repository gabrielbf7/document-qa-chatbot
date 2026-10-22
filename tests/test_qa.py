import pytest

from src.config import PRICE_INPUT_PER_MTOK, PRICE_OUTPUT_PER_MTOK
from src.qa import FALLBACK_ANSWER, build_prompt, calculate_cost, parse_response

CHUNKS = [
    {"chunk_id": "faq.md#0", "source": "faq.md", "section": "Hours", "page": None, "text": "Open Monday to Friday."},
    {"chunk_id": "services.pdf#1", "source": "services.pdf", "section": None, "page": 1, "text": "Cleaning $110."},
]


def test_build_prompt_includes_all_chunk_ids_and_question():
    prompt = build_prompt("Are you open on Sundays?", CHUNKS)
    for chunk in CHUNKS:
        assert f'id="{chunk["chunk_id"]}"' in prompt
        assert chunk["text"] in prompt
    assert prompt.rstrip().endswith("Question: Are you open on Sundays?")


def test_build_prompt_puts_question_after_documents():
    prompt = build_prompt("Q?", CHUNKS)
    assert prompt.index("</document>") < prompt.index("Question: Q?")


def test_build_prompt_escapes_attribute_quotes():
    chunk = {**CHUNKS[0], "section": 'Say "hi"'}
    assert 'section="Say "hi""' not in build_prompt("Q?", [chunk])


def test_calculate_cost():
    expected = (1000 * PRICE_INPUT_PER_MTOK + 200 * PRICE_OUTPUT_PER_MTOK) / 1_000_000
    assert calculate_cost(1000, 200) == pytest.approx(expected)
    assert calculate_cost(1_000_000, 0) == pytest.approx(PRICE_INPUT_PER_MTOK)
    assert calculate_cost(0, 0) == 0


def test_parse_valid_json_maps_sources_in_code():
    text = '{"answered": true, "answer": "Closed.", "sources": ["faq.md#0", "services.pdf#1"]}'
    result = parse_response(text, CHUNKS)
    assert result["answered"] is True
    assert result["answer"] == "Closed."
    assert result["sources"] == [
        {"chunk_id": "faq.md#0", "source": "faq.md", "section": "Hours", "page": None},
        {"chunk_id": "services.pdf#1", "source": "services.pdf", "section": None, "page": 1},
    ]


def test_parse_drops_invalid_chunk_ids():
    text = '{"answered": true, "answer": "Yes.", "sources": ["faq.md#0", "made_up.md#9"]}'
    result = parse_response(text, CHUNKS)
    assert [s["chunk_id"] for s in result["sources"]] == ["faq.md#0"]


def test_parse_answered_with_only_invalid_ids_is_not_answered():
    text = '{"answered": true, "answer": "Yes.", "sources": ["made_up.md#9"]}'
    result = parse_response(text, CHUNKS)
    assert result == {"answered": False, "answer": FALLBACK_ANSWER, "sources": []}


def test_parse_not_answered_has_no_sources():
    text = '{"answered": false, "answer": "I don\'t know.", "sources": ["faq.md#0"]}'
    result = parse_response(text, CHUNKS)
    assert result == {"answered": False, "answer": "I don't know.", "sources": []}


def test_parse_json_inside_code_fence_and_prose():
    text = 'Here you go:\n```json\n{"answered": true, "answer": "A", "sources": ["faq.md#0"]}\n```'
    assert parse_response(text, CHUNKS)["answered"] is True


@pytest.mark.parametrize(
    "text",
    ["", "not json at all", '{"answered": true, "answer": ', "[1, 2, 3]", '{"answered": true}'],
)
def test_parse_malformed_json_falls_back(text):
    result = parse_response(text, CHUNKS)
    assert result == {"answered": False, "answer": FALLBACK_ANSWER, "sources": []}
