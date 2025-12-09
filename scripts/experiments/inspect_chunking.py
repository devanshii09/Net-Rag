import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.data.chunking import (  # type: ignore
    chunk_unsw_row_based,
    chunk_unsw_session_based,
)


def main():
    df = load_unsw_nb15_full()

    print("[INFO] Sampling 2000 rows for quicker testing...")
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)

    print("\n[INFO] Row-based chunking (first 5 rows)...")
    row_chunks = chunk_unsw_row_based(df_sample.head(5))
    for c in row_chunks:
        print("ID:", c["id"])
        print("META:", c["meta"])
        print("TEXT:", c["text"])
        print("-" * 60)

    print("\n[INFO] Session-based chunking on sample of 200 rows...")
    session_chunks = chunk_unsw_session_based(df_sample.head(200))
    print(f"Total session chunks: {len(session_chunks)}")
    for c in session_chunks[:5]:
        print("ID:", c["id"])
        print("META:", c["meta"])
        print("TEXT (first 300 chars):", c["text"][:300].replace("\n", " "))
        print("-" * 60)


if __name__ == "__main__":
    main()