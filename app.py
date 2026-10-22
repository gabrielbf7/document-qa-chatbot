import streamlit as st

from src.indexer import build_index, get_model, index_exists, load_index
from src.qa import FALLBACK_ANSWER, answer, get_client
from src.uploads import save_upload

st.set_page_config(page_title="Maplewood Dental Studio Assistant", page_icon="🦷")


@st.cache_resource
def cached_model():
    return get_model()


@st.cache_resource
def cached_client():
    return get_client()


@st.cache_resource
def cached_index():
    return load_index()


def format_source(source: dict) -> str:
    parts = [source["source"], source["section"], f"p. {source['page']}" if source["page"] else None]
    return " — ".join(part for part in parts if part)


def render_body(message: dict) -> None:
    st.markdown(message["content"])
    if message["role"] == "assistant":
        if message["sources"]:
            st.caption("Sources: " + "; ".join(message["sources"]))
        st.caption(f"Cost: ${message['cost']:.5f}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        render_body(message)


def handle_uploads() -> None:
    uploaded = st.file_uploader(
        "Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    saved_now = st.session_state.setdefault("saved_uploads", set())
    new_files = 0
    for file in uploaded:
        key = (file.name, file.size)
        if key not in saved_now and save_upload(file.name, file.getvalue()):
            saved_now.add(key)
            new_files += 1
    if new_files:
        st.info(f"{new_files} file(s) saved. Click Reindex to include them.")


def reindex() -> None:
    with st.spinner("Building index..."):
        chunks = build_index()
    cached_index.clear()
    st.success(f"Indexed {len(chunks)} chunks.")


st.title("🦷 Maplewood Dental Studio Assistant")
st.write(
    "Ask about hours, services, prices and policies. "
    "Answers come only from the clinic's documents. You can write in English or Portuguese."
)

st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Documents")
    handle_uploads()
    if st.button("Reindex"):
        reindex()
    total_box = st.empty()

if not index_exists():
    st.warning("The search index has not been built yet.")
    if st.button("Build index"):
        reindex()
        st.rerun()
    st.stop()

try:
    client = cached_client()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

with st.spinner("Loading search model..."):
    cached_model()
index = cached_index()

with st.sidebar:
    st.caption(f"Index: {len(index[1])} chunks from {len({c['source'] for c in index[1]})} files")

for message in st.session_state.history:
    render_message(message)

if question := st.chat_input("Ask a question"):
    st.session_state.history.append(
        {"role": "user", "content": question, "sources": [], "cost": 0.0}
    )
    render_message(st.session_state.history[-1])
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                result = answer(question, client=client, index=index)
            except Exception as error:
                result = None
                st.session_state.history.pop()
                st.error(f"Could not get an answer ({type(error).__name__}). Please try again.")
        if result:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result["answer"] or FALLBACK_ANSWER,
                    "sources": [format_source(s) for s in result["sources"]],
                    "cost": result["cost_usd"],
                }
            )
            render_body(st.session_state.history[-1])

total_cost = sum(m["cost"] for m in st.session_state.history)
total_box.metric("Session cost (USD)", f"${total_cost:.5f}")
