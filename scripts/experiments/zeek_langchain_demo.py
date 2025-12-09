# scripts/zeek_langchain_demo.py

import argparse
from collections import Counter
from typing import Optional

from scripts.zeek_langchain_service import explain_host_behavior  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI demo: Net-RAG over Zeek logs using LangChain + Ollama + Chroma"
    )
    parser.add_argument(
        "--src-ip",
        required=True,
        help="Source IP (id.orig_h) of host to investigate.",
    )
    parser.add_argument(
        "--dest-port",
        type=int,
        default=None,
        help="Destination port (id.resp_p) to focus on (optional).",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Analyst question to ask the Net-RAG system.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max number of Zeek sessions to retrieve.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild Chroma index from Zeek logs before querying.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama server.",
    )
    return parser.parse_args()


def print_context_summary(contexts):
    origin_hosts = Counter()
    dest_hosts = Counter()
    services = Counter()

    for ctx in contexts:
        meta = ctx.get("meta", {}) or {}
        origin_hosts[meta.get("id.orig_h", "unknown")] += 1
        dest_hosts[meta.get("id.resp_h", "unknown")] += 1
        services[meta.get("service", "unknown")] += 1

    print("\n[INFO] Source doc summary (RAG context):")
    if origin_hosts:
        print("  Origin hosts:")
        for host, cnt in origin_hosts.items():
            print(f"    {host}: {cnt}")
    if dest_hosts:
        print("  Destination hosts:")
        for host, cnt in dest_hosts.items():
            print(f"    {host}: {cnt}")
    if services:
        print("  Services:")
        for svc, cnt in services.items():
            print(f"    {svc}: {cnt}")


def main() -> None:
    args = parse_args()

    result = explain_host_behavior(
        src_ip=args.src_ip,
        dest_port=args.dest_port,
        question=args.query,
        rebuild_index=args.rebuild_index,
        ollama_base_url=args.ollama_base_url,
        top_k=args.top_k,
    )

    answer: str = result.get("answer", "") or ""
    contexts = result.get("contexts", [])

    print("\n[INFO] LLM Answer:\n")
    print(answer.strip() or "[No answer returned]")

    if not contexts:
        print("\n[INFO] No Zeek sessions were retrieved for this query.")
        return

    print_context_summary(contexts)

    print("\n[INFO] Retrieved Zeek sessions (truncated):\n")
    for i, ctx in enumerate(contexts, start=1):
        uid = ctx.get("uid", f"session_{i}")
        meta = ctx.get("meta", {}) or {}
        text = ctx.get("text", "")

        print(f"  --- Context #{i} ---")
        print(f"  uid: {uid}")
        print(f"  meta: {meta}")
        print("  text:", text[:300].replace("\n", " "))
        if len(text) > 300:
            print("  ...")
        print("")


if __name__ == "__main__":
    main()