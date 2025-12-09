import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

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

# --------------------------------------------------------------------
# Imports from your project
# --------------------------------------------------------------------
from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.rag.embedding_index import (  # type: ignore
    get_embedding_model,
    get_chroma_client,
    query_similar_chunks,
)
from net_rag.data.zeek_loader import load_zeek_logs  # type: ignore
from net_rag.data.zeek_chunking import chunk_zeek_sessions  # type: ignore

from eval_unsw_baseline import train_open_set_baseline  # type: ignore
from explain_unsw_flow import (  # type: ignore
    UNSEEN_CATS,
    predict_and_explain,
    row_to_text,
    TOP_K,
    THRESHOLD,
)

# --------------------------------------------------------------------
# UNSW helpers
# --------------------------------------------------------------------


def explain_unsw_flow(
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
    and print results.
    """
    true_label = int(row["label"])
    true_cat = row["attack_cat"]
    flow_id = int(row["id"])
    in_unseen = true_cat in UNSEEN_CATS

    print("\n[INFO] Selected UNSW test flow:")
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

    print("\n[INFO] Neighbors (top 3):")
    for i, n in enumerate(expl["neighbors"][:3], start=1):
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
    print(f"  P(malicious): {baseline_proba:.3f}\n")


# --------------------------------------------------------------------
# Zeek helpers
# --------------------------------------------------------------------


def load_zeek_and_chunks():
    logs = load_zeek_logs("data/zeek_demo")
    conn_df = logs["conn"]
    http_df = logs.get("http")
    dns_df = logs.get("dns")

    chunks = chunk_zeek_sessions(conn_df, http_df, dns_df)
    if not chunks:
        print("[ERROR] No Zeek session chunks found.")
        return None, None

    # Build a quick uid->chunk and index->chunk mapping for lookups
    uid_index: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        uid = c["meta"].get("uid")
        if uid:
            uid_index[uid] = c

    return chunks, uid_index


def zeek_query_by_chunk(chunk: Dict[str, Any], model, top_k: int = 5) -> None:
    query_text = chunk["text"]
    meta = chunk["meta"]
    print("\n[INFO] Selected Zeek session for query:")
    print("  id:", chunk["id"])
    print("  meta:", meta)
    print(
        "  text:",
        query_text[:300].replace("\n", " ")
        + ("..." if len(query_text) > 300 else ""),
    )

    print("\n[INFO] Querying nearest neighbors in Chroma (Zeek sessions)...")
    results = query_similar_chunks(
        query_text,
        model,
        persist_dir="data/processed/chroma_zeek_sessions",
        collection_name="zeek_sessions",
        top_k=top_k,
    )

    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0] if "distances" in results else [
        None
    ] * len(ids_list)

    # quick aggregate summary
    from collections import Counter

    orig_hosts = Counter()
    resp_hosts = Counter()
    services = Counter()

    for m in metas_list:
        oh = m.get("id.orig_h")
        rh = m.get("id.resp_h")
        sv = m.get("service")
        if oh:
            orig_hosts[oh] += 1
        if rh:
            resp_hosts[rh] += 1
        if sv:
            services[sv] += 1

    print("\n[INFO] Neighbor summary:")
    if orig_hosts:
        print("  Origin hosts:")
        for h, c in orig_hosts.most_common():
            print(f"    {h}: {c}/{len(metas_list)}")
    if services:
        print("  Services:")
        for s, c in services.most_common():
            print(f"    {s}: {c}/{len(metas_list)}")
    if resp_hosts:
        print("  Destination hosts:")
        for h, c in resp_hosts.most_common():
            print(f"    {h}: {c}/{len(metas_list)}")

    print("\n[INFO] Retrieved neighbors:")
    for i, (rid, doc, m, dist) in enumerate(
        zip(ids_list, docs_list, metas_list, dists_list), start=1
    ):
        print(f"\nNeighbor {i}:")
        print("  id:", rid)
        print("  distance:", dist)
        print("  meta:", m)
        txt = doc.replace("\n", " ")
        if len(txt) > 300:
            txt = txt[:300] + "..."
        print("  text:", txt)


# --------------------------------------------------------------------
# Main CLI menu
# --------------------------------------------------------------------


def main() -> None:
    print("[INFO] Loading UNSW data and training baseline RF...")
    baseline = train_open_set_baseline(unseen_cats=UNSEEN_CATS)
    rf_pipe = baseline["pipeline"]
    numeric_cols = baseline["numeric_cols"]
    cat_cols = baseline["cat_cols"]

    df = baseline["df"]
    df_test = df[df["split"] == "test"].copy()
    print(f"[INFO] UNSW test set size: {len(df_test)}")

    print("\n[INFO] Loading embedding model (used for BOTH UNSW and Zeek demos)...")
    model = get_embedding_model()

    print("[INFO] Loading Zeek logs and building session chunks...")
    zeek_chunks, zeek_uid_index = load_zeek_and_chunks()
    if zeek_chunks is None:
        print("[WARN] Zeek demos will not work (no chunks).")

    # ------------------------------------------------------------
    # Simple interactive menu
    # ------------------------------------------------------------
    while True:
        print("\n================ DEMO MENU ================")
        print("1. UNSW – explain specific flow by id")
        print("2. UNSW – run three curated cases (72072, 45575, 61997)")
        print("3. Zeek – query by IP")
        print("4. Zeek – query by UID")
        print("5. Quit")
        choice = input("Select an option [1-5]: ").strip()

        if choice == "1":
            try:
                fid = int(input("Enter UNSW flow id (e.g., 72072): ").strip())
            except ValueError:
                print("[ERROR] Invalid id.")
                continue
            mask = df_test["id"] == fid
            if not mask.any():
                print(f"[ERROR] No test flow found with id={fid}.")
                continue
            row = df_test[mask].iloc[0]
            explain_unsw_flow(row, model, rf_pipe, numeric_cols, cat_cols)

        elif choice == "2":
            demo_ids = [
                (72072, "Case A: Normal, RF FP, RAG TP (benign)"),
                (45575, "Case B: Fuzzers attack, RF TP, RAG FN"),
                (61997, "Case C: Shellcode (unseen), RAG generalises"),
            ]
            for fid, title in demo_ids:
                print("\n" + "=" * 80)
                print(title)
                print("=" * 80)
                mask = df_test["id"] == fid
                if not mask.any():
                    print(f"[ERROR] No test flow found with id={fid}. Skipping.")
                    continue
                row = df_test[mask].iloc[0]
                explain_unsw_flow(row, model, rf_pipe, numeric_cols, cat_cols)

        elif choice == "3":
            if zeek_chunks is None:
                print("[ERROR] Zeek chunks not available.")
                continue
            ip = input("Enter IP (matches id.orig_h): ").strip()
            if not ip:
                print("[ERROR] Empty IP.")
                continue
            # find first chunk with this origin IP
            match = None
            for c in zeek_chunks:
                if c["meta"].get("id.orig_h") == ip:
                    match = c
                    break
            if match is None:
                print(f"[ERROR] No Zeek session found with id.orig_h={ip}.")
                continue
            zeek_query_by_chunk(match, model, top_k=10)

        elif choice == "4":
            if zeek_uid_index is None:
                print("[ERROR] Zeek chunks not available.")
                continue
            uid = input("Enter Zeek UID (e.g., uid_C4l5JE4LygdXhiKZVg): ").strip()
            if not uid:
                print("[ERROR] Empty UID.")
                continue
            chunk = zeek_uid_index.get(uid)
            if chunk is None:
                print(f"[ERROR] No Zeek session found with uid={uid}.")
                continue
            zeek_query_by_chunk(chunk, model, top_k=10)

        elif choice == "5":
            print("Exiting demo.")
            break
        else:
            print("[WARN] Invalid choice. Please enter 1–5.")


if __name__ == "__main__":
    main()