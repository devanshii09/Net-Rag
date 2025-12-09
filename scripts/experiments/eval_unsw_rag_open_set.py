import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import (  # type: ignore
    get_embedding_model,
    index_chunks,
    get_chroma_client,
)

UNSEEN_CATS = ["Worms", "Shellcode", "Backdoor"]
TOP_K = 11
THRESHOLD = 0.6  # predict malicious if >= 70% neighbors are malicious/ tested with ) 0.5 threshould


# --- Text serialization (same style as build_unsw_index.py) ---


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


def make_row_chunks(df: pd.DataFrame) -> List[Dict[str, Any]]:
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


# --- Build open-set train index (seen categories only) ---


def build_open_set_index(df: pd.DataFrame) -> None:
    """
    Build Chroma index on train split, excluding UNSEEN_CATS attack categories.
    """
    df_seen_train = df[(df["split"] == "train") & (~df["attack_cat"].isin(UNSEEN_CATS))].copy()
    print(f"[INFO] RAG open-set: train rows (seen cats): {len(df_seen_train)}")

    chunks = make_row_chunks(df_seen_train)
    print(f"[INFO] RAG open-set: chunks (seen cats): {len(chunks)}")

    model = get_embedding_model()

    index_chunks(
        chunks,
        model,
        persist_dir="data/processed/chroma_unsw_open_set",
        collection_name="unsw_open_set_seen_train",
        batch_size=512,
    )


# --- k-NN prediction using embeddings ---


def rag_knn_predict(
    texts: List[str],
    model,
    top_k: int = TOP_K,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    client = get_chroma_client("data/processed/chroma_unsw_open_set")
    collection = client.get_collection("unsw_open_set_seen_train")

    preds: List[int] = []
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    for emb in embeddings:
        res = collection.query(query_embeddings=[emb], n_results=top_k)
        metas = res["metadatas"][0]
        neighbor_labels = [m.get("label") for m in metas]
        neighbor_labels = [int(l) for l in neighbor_labels if l is not None]

        if not neighbor_labels:
            preds.append(0)
            continue

        frac_malicious = sum(neighbor_labels) / len(neighbor_labels)
        preds.append(1 if frac_malicious >= threshold else 0)

    return np.array(preds, dtype=int)


def df_to_texts(df_part: pd.DataFrame) -> List[str]:
    return [row_to_text(row) for _, row in df_part.iterrows()]


def main() -> None:
    df = load_unsw_nb15_full()

    df_test = df[df["split"] == "test"].copy()
    df_test_seen = df_test[~df_test["attack_cat"].isin(UNSEEN_CATS)].copy()
    df_test_unseen = df_test[df_test["attack_cat"].isin(UNSEEN_CATS)].copy()

    print("[INFO] Open-set config:")
    print(f"  Unseen categories: {UNSEEN_CATS}")
    print(f"  Test size:         {len(df_test)}")
    print(f"  Test seen flows:   {len(df_test_seen)}")
    print(f"  Test unseen flows: {len(df_test_unseen)}")

    # 1) Build index on seen train
    build_open_set_index(df)

    # 2) Load model once for prediction
    print("\n[INFO] Loading embedding model for prediction...")
    model = get_embedding_model()

    # --- ALL test flows ---
    texts_all = df_to_texts(df_test)
    y_all = df_test["label"].astype(int).values
    y_pred_all = rag_knn_predict(texts_all, model)

    print("\n[INFO] RAG k-NN (ALL test flows)")
    print(classification_report(y_all, y_pred_all, digits=4))
    print("Confusion matrix (all test):")
    print(confusion_matrix(y_all, y_pred_all))

    # --- SEEN categories only ---
    texts_seen = df_to_texts(df_test_seen)
    y_seen = df_test_seen["label"].astype(int).values
    y_pred_seen = rag_knn_predict(texts_seen, model)

    print("\n[INFO] RAG k-NN (SEEN categories only)")
    print(classification_report(y_seen, y_pred_seen, digits=4))
    print("Confusion matrix (seen):")
    print(confusion_matrix(y_seen, y_pred_seen))

    # --- UNSEEN categories only ---
    texts_unseen = df_to_texts(df_test_unseen)
    y_unseen = df_test_unseen["label"].astype(int).values
    y_pred_unseen = rag_knn_predict(texts_unseen, model)

    print("\n[INFO] RAG k-NN (UNSEEN categories only)")
    print(classification_report(y_unseen, y_pred_unseen, digits=4))
    print("Confusion matrix (unseen):")
    print(confusion_matrix(y_unseen, y_pred_unseen))


if __name__ == "__main__":
    main()