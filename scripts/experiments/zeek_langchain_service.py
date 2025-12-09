# scripts/zeek_langchain_service.py

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# LangChain / Chroma / Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

# Make src/ importable for net_rag modules
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.zeek_loader import load_zeek_logs  # type: ignore
from net_rag.data.zeek_chunking import chunk_zeek_sessions  # type: ignore


DATA_DIR = ROOT / "data" / "zeek_demo"
PERSIST_DIR = ROOT / "data" / "processed" / "chroma_zeek_langchain"
COLLECTION_NAME = "zeek_sessions_langchain"


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma >=0.5 only accepts simple types in metadata.
    Convert numpy scalars etc. to Python primitives / strings.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        # Convert numpy scalars to Python types
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _build_zeek_chunks(data_dir: Path) -> List[Dict[str, Any]]:
    """Load Zeek logs and build session chunks."""
    print(f"[INFO] Loading {data_dir / 'conn.log'}")
    print(f"[INFO] Loading {data_dir / 'dns.log'}")
    logs = load_zeek_logs(str(data_dir))
    conn_df = logs["conn"]
    http_df = logs.get("http")
    dns_df = logs.get("dns")

    chunks = chunk_zeek_sessions(conn_df, http_df, dns_df)
    print(f"[INFO] Built {len(chunks)} Zeek session chunks from {data_dir}")
    return chunks


def build_or_load_chroma(
    rebuild_index: bool = False,
    ollama_base_url: Optional[str] = None,
) -> Chroma:
    """
    Build (or load) a Chroma vector index of Zeek session chunks.
    Uses OllamaEmbeddings (nomic-embed-text).
    """
    print("[INFO] Initializing Ollama embedding model (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_base_url or "http://localhost:11434",
    )

    persist_dir = PERSIST_DIR
    collection_name = COLLECTION_NAME

    if rebuild_index:
        # We always rebuild from logs
        from shutil import rmtree

        if persist_dir.exists():
            rmtree(persist_dir)

        chunks = _build_zeek_chunks(DATA_DIR)
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for ch in chunks:
            cid = str(ch.get("id"))
            raw_meta = ch.get("meta", {}) or {}
            # Ensure uid is present
            raw_meta.setdefault("uid", cid)
            meta = _sanitize_metadata(raw_meta)

            texts.append(ch.get("text", ""))
            metadatas.append(meta)
            ids.append(cid)

        print(
            f"[INFO] Indexing {len(texts)} chunks into Chroma collection "
            f"'{collection_name}' at {persist_dir}"
        )
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )

        # Chroma 0.4+ auto-persists, but this keeps your earlier logs similar
        try:
            vectorstore.persist()
        except Exception:
            pass

        print("[INFO] Chroma index build complete.")
        return vectorstore

    # Load existing index
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    print("[INFO] Loaded existing Chroma index.")
    return vectorstore


def _python_filter_docs(
    docs,
    src_ip: Optional[str],
    dest_port: Optional[int],
    top_k: int,
):
    """
    Filter LangChain Documents in Python by metadata (id.orig_h, id.resp_p)
    instead of relying on Chroma's where-clauses (which are brittle across versions).
    """
    filtered = []
    for d in docs:
        meta = d.metadata or {}

        if src_ip and meta.get("id.orig_h") != src_ip:
            continue

        if dest_port is not None:
            dst_raw = meta.get("id.resp_p")
            try:
                dst_int = int(dst_raw)
            except (TypeError, ValueError):
                continue
            if dst_int != dest_port:
                continue

        filtered.append(d)
        if len(filtered) >= top_k:
            break

    # If nothing matched strictly, fall back to the original top_k docs
    if not filtered:
        filtered = list(docs[:top_k])
    return filtered


def explain_host_behavior(
    src_ip: str,
    dest_port: Optional[int],
    question: str,
    rebuild_index: bool = False,
    ollama_base_url: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Core Net-RAG function used by both the CLI demo and the Streamlit app.

    Steps:
      1. Build / load Chroma index over Zeek sessions.
      2. Retrieve similar sessions (vector search).
      3. Filter by src_ip / dest_port in Python.
      4. Call Ollama LLM (ChatOllama) with context + question.
      5. Return answer + contexts.
    """
    vectorstore = build_or_load_chroma(
        rebuild_index=rebuild_index,
        ollama_base_url=ollama_base_url,
    )

    # Enrich the question text itself so embeddings "know" the host/port
    base_question = question.strip() or "Explain this host's behavior."
    prefix_bits = []
    if src_ip:
        prefix_bits.append(f"Host: {src_ip}")
    if dest_port is not None:
        prefix_bits.append(f"Dest port: {dest_port}")
    prefix = " ".join(f"[{p}]" for p in prefix_bits)
    full_query = f"{prefix} {base_question}" if prefix else base_question

    print("\n[INFO] Running RetrievalQA-style query...")
    print(f"  Question: {full_query}")
    if src_ip:
        print(f"  Python metadata filter src_ip (id.orig_h): {src_ip}")
    if dest_port is not None:
        print(f"  Python metadata filter dest_port (id.resp_p): {dest_port}")

    # Retrieve more than we need, then filter in Python
    retriever = vectorstore.as_retriever(search_kwargs={"k": max(top_k * 3, top_k)})
    raw_docs = retriever.invoke(full_query)
    docs = _python_filter_docs(raw_docs, src_ip=src_ip, dest_port=dest_port, top_k=top_k)

    # Prepare context for the LLM
    contexts: List[Dict[str, Any]] = []
    for d in docs:
        meta = dict(d.metadata or {})
        uid = meta.get("uid") or meta.get("id") or "unknown"
        contexts.append(
            {
                "uid": uid,
                "meta": meta,
                "text": d.page_content,
            }
        )

    # Build prompt
    prompt_parts = [
        "You are a network security analyst.",
        "You are given Zeek session summaries describing network connections.",
        "Use ONLY the provided sessions as ground truth.",
        "Explain clearly what the host is doing and whether it looks normal or suspicious.",
        "Reference specific IPs, ports, and protocols in your explanation.",
        "",
        f"Host under investigation: {src_ip}",
    ]
    if dest_port is not None:
        prompt_parts.append(f"Destination port of interest: {dest_port}")
    prompt_parts.append("")
    prompt_parts.append("=== Zeek Sessions ===")

    for i, ctx in enumerate(contexts, start=1):
        prompt_parts.append(f"\n[Session {i}] uid={ctx['uid']}")
        prompt_parts.append(ctx["text"])

    prompt_parts.append("\n=== Question ===")
    prompt_parts.append(base_question)
    prompt_parts.append("\n=== Answer ===")

    prompt = "\n".join(prompt_parts)

    # Call LLM via Ollama
    llm = ChatOllama(
        model="llama3.1",  # or any model you prefer
        base_url=ollama_base_url or "http://localhost:11434",
        temperature=0.2,
    )

    answer_msg = llm.invoke(prompt)
    answer = getattr(answer_msg, "content", str(answer_msg))

    return {
        "answer": answer,
        "contexts": contexts,
    }