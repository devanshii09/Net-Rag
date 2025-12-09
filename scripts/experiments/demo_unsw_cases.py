import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# Path setup so we can import from src/ and scripts/
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse your existing utilities
from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import get_embedding_model  # type: ignore

from eval_unsw_baseline import train_open_set_baseline  # type: ignore
from explain_unsw_flow import (  # type: ignore
    UNSEEN_CATS,
    predict_and_explain,
    row_to_text,
    TOP_K,
    THRESHOLD,
)


def print_case_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def explain_flow(
    row: pd.Series,
    model,
    rf_pipe,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> None:
    """
    Given a single UNSW row, run:
      - RAG k-NN explanation
      - Baseline RF prediction
    and print results in a compact, demo-friendly format.
    """
    true_label = int(row["label"])
    true_cat = row["attack_cat"]
    flow_id = int(row["id"])
    in_unseen = true_cat in UNSEEN_CATS

    print("[INFO] Selected UNSW test flow:")
    print(f"  id: {flow_id}")
    print(f"  attack_cat: {true_cat}")
    print(f"  true_label: {true_label} ({'malicious' if true_label == 1 else 'benign'})")
    print(f"  in UNSEEN_CATS: {in_unseen}")
    print("  text:", row_to_text(row))

    # ---------- RAG prediction ----------
    print("\n[INFO] Running RAG k-NN prediction and explanation...")
    pred_label, expl = predict_and_explain(row, model, top_k=TOP_K, threshold=THRESHOLD)

    print("\n[INFO] Prediction (RAG k-NN):")
    print(
        f"  predicted_label: {pred_label} "
        f"({'malicious' if pred_label == 1 else 'benign'})"
    )
    print(
        f"  frac_malicious among {expl['top_k']} neighbors: "
        f"{expl['frac_malicious']:.3f} (threshold={expl['threshold']})"
    )

    print("\n[INFO] Neighbor attack category distribution:")
    for cat, count in sorted(expl["cat_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}/{expl['top_k']} neighbors")

    print("\n[INFO] Neighbors (top 5):")
    for i, n in enumerate(expl["neighbors"][:5], start=1):
        meta = n["meta"]
        print(f"\nNeighbor {i}:")
        print("  id:", n["id"])
        print("  distance:", n["distance"])
        print("  meta:", meta)
        txt = n["text"].replace("\n", " ")
        if len(txt) > 250:
            txt = txt[:250] + "..."
        print("  text:", txt)

    # ---------- Baseline RF prediction ----------
    row_df = row.to_frame().T
    X_row = row_df[numeric_cols + cat_cols]

    baseline_pred = rf_pipe.predict(X_row)[0]
    baseline_proba = rf_pipe.predict_proba(X_row)[0, 1]

    print("\n[INFO] Baseline RandomForest prediction:")
    print(
        f"  predicted_label: {baseline_pred} "
        f"({'malicious' if baseline_pred == 1 else 'benign'})"
    )
    print(f"  P(malicious): {baseline_proba:.3f}")


def main() -> None:
    # ----------------------------------------------------------------
    # 1. Train baseline once
    # ----------------------------------------------------------------
    print("[INFO] Training baseline RandomForest via train_open_set_baseline()...")
    baseline = train_open_set_baseline(unseen_cats=UNSEEN_CATS)
    rf_pipe = baseline["pipeline"]
    numeric_cols = baseline["numeric_cols"]
    cat_cols = baseline["cat_cols"]

    df = baseline["df"]
    df_test = df[df["split"] == "test"].copy()

    print("[INFO] Test set size:", len(df_test))

    # ----------------------------------------------------------------
    # 2. Load embedding model once
    # ----------------------------------------------------------------
    print("\n[INFO] Loading embedding model for RAG...")
    model = get_embedding_model()

    # ----------------------------------------------------------------
    # 3. Define interesting demo cases
    # ----------------------------------------------------------------
    demo_cases: List[Dict[str, Any]] = [
        {
            "id": 72072,
            "title": "Case A: Normal flow (RAG fixes RF false positive)",
        },
        {
            "id": 45575,
            "title": "Case B: Fuzzers attack (RF catches attack RAG misses)",
        },
        {
            "id": 61997,
            "title": "Case C: Unseen Shellcode attack (open-set generalisation)",
        },
    ]

    # ----------------------------------------------------------------
    # 4. Run through each case
    # ----------------------------------------------------------------
    for case in demo_cases:
        flow_id = case["id"]
        title = case["title"]

        print_case_header(title)

        mask = df_test["id"] == flow_id
        if not mask.any():
            print(f"[ERROR] No test flow found with id={flow_id}. Skipping.")
            continue

        row = df_test[mask].iloc[0]
        explain_flow(row, model, rf_pipe, numeric_cols, cat_cols)


if __name__ == "__main__":
    main()