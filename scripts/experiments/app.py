# app.py

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Make scripts/ importable
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.zeek_langchain_service import explain_host_behavior  # type: ignore


def parse_dest_port(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        port = int(raw)
        if port < 0 or port > 65535:
            return None
        return port
    except ValueError:
        return None


def main() -> None:
    st.set_page_config(
        page_title="Net-RAG Zeek Demo",
        layout="wide",
    )

    st.title("Net-RAG: Zeek + LangChain + Ollama Demo")
    st.caption(
        "Interactive demo of Retrieval-Augmented Generation over Zeek session logs. "
        "You ask a question about a host; the system retrieves matching Zeek sessions "
        "and an LLM explains what is happening."
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Query Parameters")

        src_ip = st.text_input(
            "Source IP (id.orig_h)",
            value="192.168.1.78",
            help="Internal host you want to investigate.",
        )

        dest_port_raw = st.text_input(
            "Destination Port (id.resp_p, optional)",
            value="5353",
            help="Leave empty to ignore, or set a specific port like 80, 443, 5353, etc.",
        )
        dest_port = parse_dest_port(dest_port_raw)

        question = st.text_area(
            "Analyst Question",
            value="Explain what this host is doing and whether it looks normal.",
            height=120,
            help="This will be passed to the LLM along with the retrieved Zeek sessions.",
        )

        top_k = st.slider(
            "Max Zeek sessions to retrieve",
            min_value=1,
            max_value=10,
            value=5,
        )

        rebuild_index = st.checkbox(
            "Rebuild Zeek index from logs",
            value=False,
            help=(
                "Re-read data/zeek_demo logs and rebuild the Chroma index. "
                "Use this if you changed the logs on disk."
            ),
        )

        ollama_base_url = st.text_input(
            "Ollama base URL",
            value="http://localhost:11434",
            help="Where your Ollama server is listening.",
        )

        run_btn = st.button("Run Net-RAG Analysis", type="primary")

    # Main panel explanation
    st.markdown("### How it works")
    st.markdown(
        """
1. Zeek logs (`conn.log`, `dns.log`, etc.) are grouped into **session chunks**.
2. Each session chunk is embedded and stored in a **Chroma** vector index.
3. When you ask a question about a host:
   - The system retrieves similar Zeek sessions (semantic search, filtered by `src_ip` and optionally `dest_port`).
   - Those sessions are passed to an **LLM via Ollama**.
   - The LLM explains what the host is doing and whether it looks suspicious.
"""
    )

    if run_btn:
        if not src_ip.strip():
            st.error("Please provide a source IP.")
            return

        with st.spinner("Running Net-RAG analysis over Zeek sessions..."):
            result = explain_host_behavior(
                src_ip=src_ip.strip(),
                dest_port=dest_port,
                question=question.strip(),
                rebuild_index=rebuild_index,
                ollama_base_url=ollama_base_url.strip() or None,
                top_k=top_k,
            )

        answer = result.get("answer", "")
        contexts = result.get("contexts", [])

        # LLM explanation
        st.markdown("## LLM Explanation")
        if answer:
            st.write(answer)
        else:
            st.warning("No answer returned from the LLM.")

        # Retrieved Zeek sessions
        st.markdown("## Retrieved Zeek Sessions (RAG Context)")

        if not contexts:
            st.info(
                "No Zeek sessions were returned for this query. "
                "Try removing the dest port filter or broadening the question."
            )
            return

        for i, ctx in enumerate(contexts, start=1):
            uid = ctx.get("uid", f"session_{i}")
            meta = ctx.get("meta", {}) or {}
            text = ctx.get("text", "")

            header = f"Context #{i} – uid={uid}"
            with st.expander(header, expanded=(i == 1)):
                st.markdown("**Metadata**")
                st.json(meta)

                st.markdown("**Session Narrative**")
                st.code(text, language="text")


if __name__ == "__main__":
    main()