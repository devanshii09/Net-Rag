import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.zeek_loader import load_zeek_logs  # type: ignore
from net_rag.data.zeek_chunking import chunk_zeek_sessions  # type: ignore


def main():
    logs = load_zeek_logs("data/zeek_demo")
    conn_df = logs["conn"]
    http_df = logs.get("http")
    dns_df = logs.get("dns")

    print(f"[INFO] conn.log rows: {len(conn_df)}")
    if http_df is not None:
        print(f"[INFO] http.log rows: {len(http_df)}")
    if dns_df is not None:
        print(f"[INFO] dns.log rows: {len(dns_df)}")

    chunks = chunk_zeek_sessions(conn_df, http_df, dns_df)
    print(f"[INFO] Built {len(chunks)} Zeek session chunks")

    for c in chunks[:5]:
        print("ID:", c["id"])
        print("META:", c["meta"])
        print("TEXT:", c["text"][:300].replace("\n", " ") + ("..." if len(c["text"]) > 300 else ""))
        print("-" * 60)


if __name__ == "__main__":
    main()