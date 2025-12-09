import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ⚠️ Use the same loader you used in inspect_unsw_nb15.py
from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import get_embedding_model, query_similar_chunks  # type: ignore


def row_to_text(row: pd.Series) -> str:
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

    text = (
        f"Flow from unknown_src:unknown_sport to unknown_dst:unknown_dsport "
        f"using {proto} (service {service}). "
        f"State {state}, duration {dur} seconds. "
        f"Source sent {sbytes} bytes in {spkts} packets, "
        f"destination sent {dbytes} bytes in {dpkts} packets. "
        f"Source TTL {sttl}, destination TTL {dttl}."
    )
    return text


def main() -> None:
    df = load_unsw_nb15_full()

    # take a small subset of test flows to query
    df_test = df[df["split"] == "test"].copy()
    df_sample = df_test.sample(1, random_state=42)
    row = df_sample.iloc[0]

    query_text = row_to_text(row)
    query_meta: Dict[str, Any] = {
        "id": int(row["id"]),
        "attack_cat": row["attack_cat"],
        "label": int(row["label"]),
    }

    print("[INFO] Selected UNSW test flow for query:")
    print("  meta:", query_meta)
    print("  text:", query_text)

    print("\n[INFO] Loading embedding model...")
    model = get_embedding_model()

    print("[INFO] Querying nearest neighbors in Chroma (UNSW train rows)...")
    results = query_similar_chunks(
        query_text,
        model,
        persist_dir="data/processed/chroma_unsw_row",
        collection_name="unsw_row_chunks",
        top_k=5,
    )

    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0] if "distances" in results else [None] * len(
        ids_list
    )

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
            doc[:250].replace("\n", " ")
            + ("..." if len(doc) > 250 else ""),
        )


if __name__ == "__main__":
    main()