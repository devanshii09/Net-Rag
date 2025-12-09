import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import get_embedding_model, get_chroma_client  # type: ignore

UNSEEN_CATS = ["Worms", "Shellcode", "Backdoor"]


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


def make_texts(df: pd.DataFrame) -> List[str]:
    return [row_to_text(row) for _, row in df.iterrows()]


def rag_knn_predict(
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
    collection_name: str = "unsw_open_set_seen_train",
) -> np.ndarray:
    client = get_chroma_client("data/processed/chroma_unsw_open_set")
    collection = client.get_collection(collection_name)

    preds = []
    for emb in embeddings:
        res = collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
        metas = res["metadatas"][0]
        neighbor_labels = [int(m.get("label", 0)) for m in metas if m.get("label") is not None]
        if not neighbor_labels:
            preds.append(0)
            continue
        frac_malicious = sum(neighbor_labels) / len(neighbor_labels)
        preds.append(1 if frac_malicious >= threshold else 0)

    return np.array(preds, dtype=int)


def evaluate_for_setting(
    y_true: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
    label: str,
) -> Tuple[float, float]:
    y_pred = rag_knn_predict(embeddings, top_k=top_k, threshold=threshold)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {label} | K={top_k}, thresh={threshold:.2f} ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, f1


def main() -> None:
    df = load_unsw_nb15_full()
    df_test = df[df["split"] == "test"].copy()
    df_test_seen = df_test[~df_test["attack_cat"].isin(UNSEEN_CATS)].copy()
    df_test_unseen = df_test[df_test["attack_cat"].isin(UNSEEN_CATS)].copy()

    print("[INFO] Test sizes:")
    print(f"  All:   {len(df_test)}")
    print(f"  Seen:  {len(df_test_seen)}")
    print(f"  Unseen:{len(df_test_unseen)}")

    print("\n[INFO] Loading embedding model...")
    model = get_embedding_model()

    print("[INFO] Encoding test splits...")
    texts_all = make_texts(df_test)
    texts_seen = make_texts(df_test_seen)
    texts_unseen = make_texts(df_test_unseen)

    emb_all = model.encode(texts_all, show_progress_bar=True)
    emb_seen = model.encode(texts_seen, show_progress_bar=True)
    emb_unseen = model.encode(texts_unseen, show_progress_bar=True)

    y_all = df_test["label"].astype(int).values
    y_seen = df_test_seen["label"].astype(int).values
    y_unseen = df_test_unseen["label"].astype(int).values

    ks = [3, 5, 11, 21]
    thresholds = [0.4, 0.5, 0.6, 0.7]

    print("\n[INFO] Sweeping over K and threshold...")
    results = []

    for k in ks:
        for t in thresholds:
            acc_all, f1_all = evaluate_for_setting(
                y_all, emb_all, k, t, label="ALL test"
            )
            acc_seen, f1_seen = evaluate_for_setting(
                y_seen, emb_seen, k, t, label="SEEN only"
            )
            acc_unseen, f1_unseen = evaluate_for_setting(
                y_unseen, emb_unseen, k, t, label="UNSEEN only"
            )

            results.append(
                {
                    "K": k,
                    "threshold": t,
                    "acc_all": acc_all,
                    "f1_all": f1_all,
                    "acc_seen": acc_seen,
                    "f1_seen": f1_seen,
                    "acc_unseen": acc_unseen,
                    "f1_unseen": f1_unseen,
                }
            )

    print("\n[SUMMARY] RAG k-NN sweep:")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()