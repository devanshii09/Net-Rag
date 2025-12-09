import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ---------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_unsw_baseline import train_open_set_baseline  # type: ignore
from net_rag.rag.embedding_index import (  # type: ignore
    get_embedding_model,
    get_chroma_client,
)

UNSEEN_CATS = ["Worms", "Shellcode", "Backdoor"]
TOP_K = 11
THRESHOLD = 0.6


# --- same serialization as other scripts ------------------------------------

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


def df_to_texts(df: pd.DataFrame) -> List[str]:
    return [row_to_text(row) for _, row in df.iterrows()]


# --- RAG k-NN prediction -----------------------------------------------------

def rag_knn_predict(
    embeddings: np.ndarray,
    top_k: int = TOP_K,
    threshold: float = THRESHOLD,
    collection_name: str = "unsw_open_set_seen_train",
) -> np.ndarray:
    client = get_chroma_client("data/processed/chroma_unsw_open_set")
    collection = client.get_collection(collection_name)

    preds: List[int] = []
    for emb in embeddings:
        res = collection.query(query_embeddings=[emb.tolist()], n_results=top_k)
        metas = res["metadatas"][0]
        neighbor_labels = [m.get("label") for m in metas]
        neighbor_labels = [int(l) for l in neighbor_labels if l is not None]

        if not neighbor_labels:
            preds.append(0)
            continue

        frac_malicious = sum(neighbor_labels) / len(neighbor_labels)
        preds.append(1 if frac_malicious >= threshold else 0)

    return np.array(preds, dtype=int)


def main() -> None:
    # -------------------------------------------------------------
    # 1. Train baseline RF and get test splits from it
    # -------------------------------------------------------------
    print("[INFO] Training baseline RandomForest via train_open_set_baseline()...")
    baseline = train_open_set_baseline(unseen_cats=UNSEEN_CATS)
    rf_pipe = baseline["pipeline"]
    numeric_cols = baseline["numeric_cols"]
    cat_cols = baseline["cat_cols"]

    df_test = baseline["df_test"]  # full test split
    df_test_seen = baseline["df_test_seen"]
    df_test_unseen = baseline["df_test_unseen"]

    print("[INFO] Test sizes (from baseline):")
    print(f"  ALL:    {len(df_test)}")
    print(f"  SEEN:   {len(df_test_seen)}")
    print(f"  UNSEEN: {len(df_test_unseen)}")

    # -------------------------------------------------------------
    # 2. Prepare data for both models
    # -------------------------------------------------------------
    # Baseline RF features
    X_test_rf = df_test[numeric_cols + cat_cols]
    y_true = df_test["label"].astype(int).values

    # RAG text & embeddings
    print("[INFO] Loading embedding model...")
    model = get_embedding_model()
    print("[INFO] Encoding ALL test flows for RAG...")
    texts_all = df_to_texts(df_test)
    emb_all = model.encode(texts_all, show_progress_bar=True)

    # -------------------------------------------------------------
    # 3. Predict with both models
    # -------------------------------------------------------------
    print("[INFO] Predicting with baseline RandomForest...")
    rf_pred = rf_pipe.predict(X_test_rf)

    print("[INFO] Predicting with RAG k-NN...")
    rag_pred = rag_knn_predict(emb_all, top_k=TOP_K, threshold=THRESHOLD)

    # sanity check
    assert len(y_true) == len(rf_pred) == len(rag_pred)

    # -------------------------------------------------------------
    # 4. Global metrics for both models
    # -------------------------------------------------------------
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    print("\n=== Baseline RF (ALL test) ===")
    print(classification_report(y_true, rf_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, rf_pred))

    print("\n=== RAG k-NN (ALL test) ===")
    print(classification_report(y_true, rag_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, rag_pred))

    print("\n[SUMMARY] Overall metrics:")
    print(f"  RF  - acc={accuracy_score(y_true, rf_pred):.4f}, "
          f"f1={f1_score(y_true, rf_pred):.4f}")
    print(f"  RAG - acc={accuracy_score(y_true, rag_pred):.4f}, "
          f"f1={f1_score(y_true, rag_pred):.4f}")

    # -------------------------------------------------------------
    # 5. Build comparison DataFrame
    # -------------------------------------------------------------
    comp = df_test[["id", "attack_cat", "label"]].copy()
    comp["rf_pred"] = rf_pred
    comp["rag_pred"] = rag_pred

    comp["seen_flag"] = ~comp["attack_cat"].isin(UNSEEN_CATS)

    def case_type(row):
        if row["rf_pred"] == row["label"] and row["rag_pred"] == row["label"]:
            return "both_correct"
        if row["rf_pred"] != row["label"] and row["rag_pred"] != row["label"]:
            return "both_wrong"
        if row["rf_pred"] == row["label"] and row["rag_pred"] != row["label"]:
            return "rf_only_correct"
        if row["rf_pred"] != row["label"] and row["rag_pred"] == row["label"]:
            return "rag_only_correct"
        return "unknown"

    comp["case_type"] = comp.apply(case_type, axis=1)

    # -------------------------------------------------------------
    # 6. Print some counts & save CSV
    # -------------------------------------------------------------
    print("\n[INFO] Case type counts (ALL test):")
    print(comp["case_type"].value_counts())

    print("\n[INFO] Case type counts by seen/unseen:")
    print(comp.groupby(["seen_flag", "case_type"]).size())

    out_path = ROOT / "data" / "processed" / "unsw_rag_vs_baseline_cases.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved per-flow comparison to: {out_path}")


if __name__ == "__main__":
    main()