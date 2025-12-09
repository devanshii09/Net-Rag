import sys
import argparse
from collections import Counter
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.zeek_loader import load_zeek_logs  # type: ignore
from net_rag.data.zeek_chunking import chunk_zeek_sessions  # type: ignore
from net_rag.rag.embedding_index import (  # type: ignore
    get_embedding_model,
    query_similar_chunks,
)


def pick_query_chunk(chunks, uid: str | None, ip: str | None):
    """
    Pick a Zeek session chunk to query.

    Priority:
      1. If uid is given -> exact match on meta['uid'].
      2. Else if ip is given -> any chunk where id.orig_h or id.resp_h == ip,
         preferring service == 'dns' if possible.
      3. Else -> first 'dns' chunk if exists, else first chunk.
    """
    if not chunks:
        raise ValueError("No Zeek chunks available")

    # 1) By UID
    if uid is not None:
        for c in chunks:
            if c["meta"].get("uid") == uid:
                return c
        raise ValueError(f"No Zeek session found with uid={uid}")

    # 2) By IP
    if ip is not None:
        ip_matches = []
        for c in chunks:
            meta = c["meta"]
            orig = meta.get("id.orig_h")
            resp = meta.get("id.resp_h")
            if orig == ip or resp == ip:
                ip_matches.append(c)

        if not ip_matches:
            raise ValueError(f"No Zeek sessions found involving IP {ip}")

        dns_like = [c for c in ip_matches if (c["meta"].get("service") or "") == "dns"]
        return dns_like[0] if dns_like else ip_matches[0]

    # 3) Default: prefer DNS, else first
    dns_like = [c for c in chunks if (c["meta"].get("service") or "") == "dns"]
    return dns_like[0] if dns_like else chunks[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Zeek RAG index with a single Zeek session."
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Specific Zeek connection UID to query.",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Restrict query to sessions involving this IP (orig or resp).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve from the index.",
    )
    args = parser.parse_args()

    # 1. Load Zeek logs
    logs = load_zeek_logs("data/zeek_demo")
    conn_df = logs["conn"]
    http_df = logs.get("http")
    dns_df = logs.get("dns")

    # 2. Build session chunks
    chunks = chunk_zeek_sessions(conn_df, http_df, dns_df)
    if not chunks:
        print("[ERROR] No Zeek session chunks found.")
        return

    # 3. Pick which session to query
    query_chunk = pick_query_chunk(chunks, uid=args.uid, ip=args.ip)

    print("[INFO] Selected Zeek session for query:")
    print("  id:", query_chunk["id"])
    print("  meta:", query_chunk["meta"])
    print(
        "  text:",
        query_chunk["text"][:300].replace("\n", " ")
        + ("..." if len(query_chunk["text"]) > 300 else ""),
    )

    query_text = query_chunk["text"]

    # 4. Load embedding model
    print("\n[INFO] Loading embedding model...")
    model = get_embedding_model()

    # 5. Query Chroma index
    print("[INFO] Querying nearest neighbors in Chroma (Zeek sessions)...")
    results = query_similar_chunks(
        query_text,
        model,
        persist_dir="data/processed/chroma_zeek_sessions",
        collection_name="zeek_sessions",
        top_k=args.top_k,
    )

    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0] if "distances" in results else [
        None
    ] * len(ids_list)

    if not ids_list:
        print("[WARN] No neighbors returned from Chroma.")
        return

    # 6. Neighbor summary (for demo)
    print("\n[INFO] Neighbor summary:")
    orig_ips = [m.get("id.orig_h", "-") for m in metas_list]
    resp_ips = [m.get("id.resp_h", "-") for m in metas_list]
    services = [m.get("service", "-") for m in metas_list]

    print("  Origin hosts:")
    for ip, c in Counter(orig_ips).most_common():
        print(f"    {ip}: {c}/{len(metas_list)}")

    print("  Services:")
    for svc, c in Counter(services).most_common():
        print(f"    {svc}: {c}/{len(metas_list)}")

    print("  Destination hosts:")
    for ip, c in Counter(resp_ips).most_common():
        print(f"    {ip}: {c}/{len(metas_list)}")

    # 7. Detailed neighbors
    print("\n[INFO] Retrieved neighbors:")
    for i, (rid, doc, meta, dist) in enumerate(
        zip(ids_list, docs_list, metas_list, dists_list), start=1
    ):
        print(f"\nNeighbor {i}:")
        print("  id:", rid)
        print("  distance:", dist)
        print("  meta:", meta)
        print(
            "  text:",
            doc[:300].replace("\n", " ")
            + ("..." if len(doc) > 300 else ""),
        )


if __name__ == "__main__":
    main()