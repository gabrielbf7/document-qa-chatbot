import json
import os
import re
import sys
from xml.sax.saxutils import quoteattr

from src.config import (
    LLM_EFFORT,
    LLM_MODEL,
    MAX_OUTPUT_TOKENS,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    ROOT_DIR,
    TOP_K,
)

FALLBACK_ANSWER = (
    "I don't know based on the clinic's documents. "
    "Please contact the clinic's front desk for help."
)

SYSTEM_PROMPT = """You are the virtual assistant of Maplewood Dental Studio, a dental clinic.
You answer patient questions using ONLY the documents provided inside <document> tags.

Rules:
- Use only information written in the documents. Do not use outside knowledge.
- Reply in the same language as the question.
- If the documents do not contain the answer, say you don't know and suggest contacting the clinic's front desk. In that case set "answered" to false and "sources" to [].
- Never give medication doses or medical advice that is not written in the documents. For medication questions, suggest contacting the clinic.
- Be concise and accurate.

Reply with a single JSON object and nothing else, in this exact shape:
{"answered": true or false, "answer": "your reply", "sources": ["chunk_id of each document you used"]}
Use the document ids exactly as given in the id attribute."""


def get_client():
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=api_key)


def build_prompt(question: str, chunks: list[dict]) -> str:
    documents = []
    for chunk in chunks:
        attrs = " ".join(
            f"{name}={quoteattr(str(value if value is not None else ''))}"
            for name, value in (
                ("id", chunk["chunk_id"]),
                ("source", chunk["source"]),
                ("section", chunk["section"]),
                ("page", chunk["page"]),
            )
        )
        documents.append(f"<document {attrs}>\n{chunk['text']}\n</document>")
    return "\n\n".join(documents) + f"\n\nQuestion: {question}"


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * PRICE_INPUT_PER_MTOK + output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in text, tolerating code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_response(text: str, chunks: list[dict]) -> dict:
    """Return {"answered", "answer", "sources"}; sources are mapped from retrieved chunks only.

    Unparseable output, or an answer citing no retrieved chunk, counts as not answered.
    """
    not_answered = {"answered": False, "answer": FALLBACK_ANSWER, "sources": []}
    data = extract_json(text)
    if data is None or not isinstance(data.get("answer"), str):
        return not_answered

    if data.get("answered") is not True:
        return {"answered": False, "answer": data["answer"], "sources": []}

    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    cited = data.get("sources")
    valid_ids = []
    for chunk_id in cited if isinstance(cited, list) else []:
        if chunk_id in by_id and chunk_id not in valid_ids:
            valid_ids.append(chunk_id)
    if not valid_ids:
        return not_answered

    sources = [
        {
            "chunk_id": chunk_id,
            "source": by_id[chunk_id]["source"],
            "section": by_id[chunk_id]["section"],
            "page": by_id[chunk_id]["page"],
        }
        for chunk_id in valid_ids
    ]
    return {"answered": True, "answer": data["answer"], "sources": sources}


def answer(question: str, client=None, index=None) -> dict:
    from src.indexer import search

    client = client or get_client()
    chunks = search(question, TOP_K, index=index)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        output_config={"effort": LLM_EFFORT},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    result = parse_response(text, chunks)
    result["input_tokens"] = response.usage.input_tokens
    result["output_tokens"] = response.usage.output_tokens
    result["cost_usd"] = calculate_cost(result["input_tokens"], result["output_tokens"])
    return result


EVAL_QUESTIONS = [
    ("Are you open on Sundays?", True),
    ("How much is a teeth cleaning?", True),
    ("Do you offer dental implants?", False),
    ("Do you offer Invisalign?", False),
    ("How much ibuprofen should I take after an extraction?", False),
    ("Vocês abrem no domingo?", True),
    ("Quanto custa uma limpeza?", True),
    ("Vocês fazem implante dentário?", False),
    ("Vocês fazem Invisalign?", False),
    ("Quanto de ibuprofeno devo tomar depois de uma extração?", False),
]


def run_eval() -> None:
    client = get_client()
    print(f"{'Question':<58} {'answered':<9} {'ok':<4} {'sources':<44} cost_usd")
    total = 0.0
    for question, expected in EVAL_QUESTIONS:
        result = answer(question, client)
        total += result["cost_usd"]
        ids = ", ".join(s["chunk_id"] for s in result["sources"]) or "-"
        ok = "OK" if result["answered"] == expected else "FAIL"
        print(f"{question:<58} {str(result['answered']):<9} {ok:<4} {ids:<44} {result['cost_usd']:.6f}")
    print(f"\nTotal cost: ${total:.6f}")


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if argv and argv[0] == "ask" and len(argv) >= 2:
        result = answer(" ".join(argv[1:]))
        print(result["answer"])
        for source in result["sources"]:
            print(f"  source: {source['chunk_id']} [{source['section']}] page={source['page']}")
        print(
            f"answered={result['answered']} tokens={result['input_tokens']}/{result['output_tokens']} "
            f"cost=${result['cost_usd']:.6f}"
        )
        return 0
    if argv and argv[0] == "eval":
        run_eval()
        return 0
    print('Usage: python -m src.qa ask "question" | eval')
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
