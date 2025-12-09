import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

# ---------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# allow importing from scripts (for eval_unsw_baseline)
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_unsw_baseline import train_open_set_baseline  # type: ignore
from net_rag.rag.embedding_index import (  # type: ignore
    get_embedding_model,
    get_chroma_client,
)

# ---------------------------------------------------------------------
# Open-set config and RAG hyperparams
# ---------------------------------------------------------------------

UNSEEN_CATS = ["Worms", "Shellcode", "Backdoor"]
TOP_K = 11
THRESHOLD = 0.6  # chosen from your sweep


# ---------------------------------------------------------------------
# Text serialization (MUST match eval scripts)
# ---------------------------------------------------------------------

def row_to_text(row: pd.Series) -> str:
    """Label-free serialization. MUST match eval scripts."""
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


# ---------------------------------------------------------------------
# RAG k-NN prediction + explanation
# ---------------------------------------------------------------------

def predict_and_explain(
    row: pd.Series,
    model,
    top_k: int = TOP_K,
    threshold: float = THRESHOLD,
) -> Tuple[int, Dict[str, Any]]:
    """
    Run k-NN in Chroma and return:
      - predicted label (0/1)
      - explanation dict with neighbor stats
    """
    client = get_chroma_client("data/processed/chroma_unsw_open_set")
    collection = client.get_collection("unsw_open_set_seen_train")

    query_text = row_to_text(row)
    query_emb = model.encode([query_text], show_progress_bar=False).tolist()[0]

    res = collection.query(query_embeddings=[query_emb], n_results=top_k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]

    neighbor_labels = [int(m.get("label", 0)) for m in metas]
    neighbor_cats = [m.get("attack_cat", "Unknown") for m in metas]

    frac_malicious = (
        sum(neighbor_labels) / len(neighbor_labels) if neighbor_labels else 0.0
    )
    pred_label = 1 if frac_malicious >= threshold else 0

    # Aggregate category counts for neighbors
    cat_counts: Dict[str, int] = {}
    for c in neighbor_cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    explanation = {
        "query_text": query_text,
        "neighbors": [
            {
                "id": ids[i],
                "distance": dists[i],
                "meta": metas[i],
                "text": docs[i],
            }
            for i in range(len(ids))
        ],
        "neighbor_labels": neighbor_labels,
        "neighbor_cats": neighbor_cats,
        "frac_malicious": frac_malicious,
        "cat_counts": cat_counts,
        "top_k": top_k,
        "threshold": threshold,
    }

    return pred_label, explanation


# ---------------------------------------------------------------------
# Row selection (seen / unseen / specific id)
# ---------------------------------------------------------------------

def pick_row(
    df_test: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    """Pick a specific test flow based on CLI args."""
    if args.seen_only and args.unseen_only:
        raise ValueError("Cannot pass both --seen-only and --unseen-only.")

    subset = df_test

    if args.seen_only:
        subset = subset[~subset["attack_cat"].isin(UNSEEN_CATS)]
    elif args.unseen_only:
        subset = subset[subset["attack_cat"].isin(UNSEEN_CATS)]

    if args.id is not None:
        # pick by UNSW flow id
        mask = subset["id"] == args.id
        if not mask.any():
            raise ValueError(
                f"No test flow found with id={args.id} in the selected subset."
            )
        return subset[mask].iloc[0]

    # default: random sample from subset
    return subset.sample(1, random_state=args.random_state).iloc[0]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain a single UNSW-NB15 flow via RAG k-NN neighbors + RF baseline."
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="UNSW flow id from CSV (test split). If omitted, sample randomly.",
    )
    parser.add_argument(
        "--seen-only",
        action="store_true",
        help="Restrict random sampling to SEEN categories only.",
    )
    parser.add_argument(
        "--unseen-only",
        action="store_true",
        help="Restrict random sampling to UNSEEN_CATS only.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling (when --id is not provided).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # 1) Train / load baseline RF on seen categories
    # -----------------------------------------------------------------
    print("\n[INFO] Training baseline RandomForest on seen categories...")
    baseline = train_open_set_baseline(unseen_cats=UNSEEN_CATS)
    rf_pipe = baseline["pipeline"]
    numeric_cols = baseline["numeric_cols"]
    cat_cols = baseline["cat_cols"]
    df_test = baseline["df_test"]  # test split from baseline script

    # -----------------------------------------------------------------
    # 2) Pick a test row (seen / unseen / specific id)
    # -----------------------------------------------------------------
    row = pick_row(df_test, args)

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

    # -----------------------------------------------------------------
    # 3) RAG k-NN prediction + explanation
    # -----------------------------------------------------------------
    print("\n[INFO] Loading embedding model...")
    model = get_embedding_model()

    print("[INFO] Running RAG k-NN prediction and explanation...")
    pred_label, expl = predict_and_explain(row, model)

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

    # -----------------------------------------------------------------
    # 4) Baseline RandomForest on the same flow
    # -----------------------------------------------------------------
    row_df = row.to_frame().T  # single-row DataFrame

    # Use the same feature subset as in baseline training
    X_row = row_df[numeric_cols + cat_cols]

    baseline_pred = rf_pipe.predict(X_row)[0]
    baseline_proba = rf_pipe.predict_proba(X_row)[0, 1]

    print("\n[INFO] Baseline RandomForest prediction:")
    print(
        f"  predicted_label: {baseline_pred} "
        f"({'malicious' if baseline_pred == 1 else 'benign'})"
    )
    print(f"  P(malicious): {baseline_proba:.3f}")


if __name__ == "__main__":
    main()