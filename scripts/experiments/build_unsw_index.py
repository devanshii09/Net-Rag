import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import get_embedding_model, index_chunks  # type: ignore


def row_to_text(row: pd.Series) -> str:
    """
    Serialize a single UNSW-NB15 flow into a narrative sentence.

    This mirrors what you saw in inspect_chunking.py:
    'Flow from unknown_src:unknown_sport to unknown_dst:unknown_dsport using udp (service dns)...'
    """
    proto = row.get("proto", "-")
    service = row.get("service", "-")
    state = row.get("state", "-")
    dur = row.get("dur", "-")

    spkts = row.get("spkts", "-")
    dpkts = row.get("dpkts", "-")
    sbytes = row.get("sbytes", "-")
    dbytes = row.get("dbytes", "-")
    sttl = row.get("sttl", "-")
    dttl = row.get("dttl", "-")

    attack_cat = row.get("attack_cat", "Unknown")
    label = row.get("label", None)
    label_str = "malicious" if label == 1 else "benign"

    text = (
        f"Flow from unknown_src:unknown_sport to unknown_dst:unknown_dsport "
        f"using {proto} (service {service}). "
        f"State {state}, duration {dur} seconds. "
        f"Source sent {sbytes} bytes in {spkts} packets, "
        f"destination sent {dbytes} bytes in {dpkts} packets. "
        f"Source TTL {sttl}, destination TTL {dttl}."
    )
    return text


def make_row_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build one chunk per UNSW row: {id, text, meta}.
    """
    chunks: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        text = row_to_text(row)
        meta: Dict[str, Any] = {
            "row_index": int(idx),
            "id": int(row.get("id", idx)),
            "proto": row.get("proto", None),
            "service": row.get("service", None),
            "state": row.get("state", None),
            "label": int(row.get("label", 0)),
            "attack_cat": row.get("attack_cat", None),
            "split": row.get("split", None),
        }

        chunks.append(
            {
                "id": f"row_{idx}",
                "text": text,
                "meta": meta,
            }
        )

    return chunks


def main() -> None:
    # 1) Load full UNSW dataset (same function you used in inspect_unsw_nb15.py)
    df = load_unsw_nb15_full()

    # use only train split for the index
    df_train = df[df["split"] == "train"].copy()
    print(f"[INFO] UNSW train rows: {len(df_train)}")

    # optional: while debugging, sample a subset
    # df_train = df_train.sample(20000, random_state=42)

    chunks = make_row_chunks(df_train)
    print(f"[INFO] Built {len(chunks)} UNSW train chunks")

    # 3) Load embedding model
    print("[INFO] Loading embedding model...")
    model = get_embedding_model()

    # 4) Index into Chroma
    index_chunks(
        chunks,
        model,
        persist_dir="data/processed/chroma_unsw_row",
        collection_name="unsw_row_chunks",
    )


if __name__ == "__main__":
    main()