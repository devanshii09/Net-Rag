import streamlit as st
import pandas as pd
import numpy as np

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# Import your existing core logic from the demo script
from unsw_open_set_rag_demo import (
    UNSW_TRAIN_CSV,
    UNSW_TEST_CSV,
    UNKNOWN_FAMILIES,
    load_unsw_train_test,
    make_open_set_splits,
    train_binary_rf,
    train_family_rf,
    build_or_load_chroma,
    rf_forward,
    family_forward,
    retrieve_neighbors,
    serialize_unsw_flow,
    compute_open_set_for_row,
    ground_truth_open_set_label,
    llm_explain_flow,
)

st.set_page_config(
    page_title="UNSW Open-set Net-RAG + LLM Demo",
    layout="wide",
)

MAX_DEBUG_SAMPLE_TRIES = 500  # how many times to try to find a misclassified flow


# -----------------------------
# Cached pipeline init
# -----------------------------
@st.cache_resource(show_spinner=True)
def init_pipeline():
    """
    Build everything once and cache:
      - Load UNSW CSVs
      - Make open-set splits
      - Train RF (binary) + family RF
      - Build/load Chroma index
    """

    df_train_raw, df_test_raw = load_unsw_train_test(UNSW_TRAIN_CSV, UNSW_TEST_CSV)

    (
        df_train_known,
        df_val_known,
        df_test_known,
        df_test_unknown,
        feature_cols,
    ) = make_open_set_splits(
        df_train_raw,
        df_test_raw,
        UNKNOWN_FAMILIES,
    )

    # Train binary RF on known families
    rf = train_binary_rf(df_train_known, df_val_known, feature_cols)

    # Train family RF (malicious-only)
    family_clf, family_le, family_classes = train_family_rf(
        df_train_known,
        feature_cols,
    )

    # Build Chroma index (Net-RAG) over train-known flows
    embedding = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )
    vectorstore = build_or_load_chroma(df_train_known, embedding)

    return {
        "df_train_known": df_train_known,
        "df_val_known": df_val_known,
        "df_test_known": df_test_known,
        "df_test_unknown": df_test_unknown,
        "feature_cols": feature_cols,
        "rf": rf,
        "family_clf": family_clf,
        "family_le": family_le,
        "family_classes": family_classes,
        "vectorstore": vectorstore,
    }


# -----------------------------
# Helper to run full pipeline on one row
# -----------------------------
def analyze_single_flow(row: pd.Series, df_name: str, pipeline: dict):
    rf = pipeline["rf"]
    family_clf = pipeline["family_clf"]
    family_le = pipeline["family_le"]
    feature_cols = pipeline["feature_cols"]
    vectorstore = pipeline["vectorstore"]

    # 1. Run RAG Retrieval FIRST (Crucial for the logic order)
    q_text = serialize_unsw_flow(row, include_labels=True)
    rag = retrieve_neighbors(vectorstore, q_text, k=5)

    # 2. Compute Open Set Label (Unpacking 5 values now)
    open_set_label, reasons, rf_out, family_out, derived_family = compute_open_set_for_row(
        rf,
        family_clf,
        family_le,
        row,
        feature_cols,
        rag_summary=rag, # Pass RAG here
    )

    # 3. LLM: explanation + decision + action
    # Note: We re-initialize ChatOllama here to ensure fresh state, or use a cached one
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    # Pass derived_family to the LLM function
    llm_text, final_decision, final_action, _ = llm_explain_flow(
        llm,
        row,
        rf_out,
        family_out,
        rag,
        open_set_label,
        derived_family, # <--- Added argument
    )

    # Ground-truth open-set label
    gt_label = ground_truth_open_set_label(row)
    family_true = str(row.get("attack_cat", "Unknown"))
    label_binary = int(row.get("label", 0))

    return {
        "df_name": df_name,
        "row_index": int(row.name),
        "row": row,
        "rf_out": rf_out,
        "family_out": family_out,
        "rag": rag,
        "open_set_label": open_set_label,
        "derived_family": derived_family, # <--- Save this for the UI
        "open_set_reasons": reasons,
        "llm_text": llm_text,
        "final_decision": final_decision,
        "final_action": final_action,
        "gt_label": gt_label,
        "family_true": family_true,
        "label_binary": label_binary,
    }


# -----------------------------
# Sampling helper for debugging errors
# -----------------------------
def sample_row_for_random(
    label_type: str,
    pipeline: dict,
    df_test_known: pd.DataFrame,
    df_test_unknown: pd.DataFrame,
    error_only: bool,
):
    """
    Sample a row for a given *ground-truth* open-set label.

    If error_only=True, keep resampling until we find a row where
    model_open_set_label != ground_truth_open_set_label, or we hit
    MAX_DEBUG_SAMPLE_TRIES and fall back to a correct row.
    """
    rf = pipeline["rf"]
    family_clf = pipeline["family_clf"]
    family_le = pipeline["family_le"]
    feature_cols = pipeline["feature_cols"]

    # Choose candidate dataframe based on GT open-set type
    if label_type == "UNKNOWN_MALICIOUS":
        df_candidates = df_test_unknown[df_test_unknown["label"] == 1]
        df_name = "test_unknown"
    elif label_type == "KNOWN_BENIGN":
        df_candidates = df_test_known[df_test_known["label"] == 0]
        df_name = "test_known"
    else:  # KNOWN_MALICIOUS
        df_candidates = df_test_known[df_test_known["label"] == 1]
        df_name = "test_known"

    if df_candidates.empty:
        return None, None

    last_row = None

    for _ in range(MAX_DEBUG_SAMPLE_TRIES):
        row = df_candidates.sample(n=1).iloc[0]
        last_row = row

        if not error_only:
            # Any row with the desired GT label is fine
            return row, df_name

        gt_label = ground_truth_open_set_label(row)
        model_label, _, _, _ = compute_open_set_for_row(
            rf,
            family_clf,
            family_le,
            row,
            feature_cols,
        )

        # We want model mistakes for debugging
        if model_label != gt_label:
            return row, df_name

    # If we didn’t find a misclassified one, fall back to the last sampled row
    st.warning(
        f"No misclassified sample found for GT label **{label_type}** "
        f"after {MAX_DEBUG_SAMPLE_TRIES} attempts. "
        "Showing a correctly classified sample instead."
    )
    return last_row, df_name


# -----------------------------
# UI helpers
# -----------------------------
def label_text_from_binary(lbl: int) -> str:
    return "benign (0)" if lbl == 0 else "malicious (1)"


def render_neighbor_table(rag_summary):
    if not rag_summary.neighbors:
        st.info("No similar historical flows retrieved from Chroma.")
        return

    rows = []
    for n in rag_summary.neighbors:
        rows.append(
            {
                "family": n.family,
                "label": "malicious" if n.label == 1 else "benign",
                "similarity": float(n.score),
                "row_idx": n.row_idx,
            }
        )
    df_neighbors = pd.DataFrame(rows)
    st.dataframe(df_neighbors, use_container_width=True)


def render_flow_summary(result: dict):
    row = result["row"]
    gt_label = result["gt_label"]
    family_true = result["family_true"]
    label_binary = result["label_binary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", result["df_name"])
    with col2:
        st.metric("UNSW row index", result["row_index"])
    with col3:
        st.metric("GT open-set label", gt_label)
    with col4:
        st.metric("GT family", family_true)

    st.caption(f"Ground-truth binary label: **{label_text_from_binary(label_binary)}**.")

    st.subheader("Serialized flow (no labels)")
    st.code(serialize_unsw_flow(row, include_labels=False))


def render_model_outputs(result: dict):
    rf_out = result["rf_out"]
    family_out = result["family_out"]
    open_set_label = result["open_set_label"]
    derived_family = result.get("derived_family", "N/A")
    reasons = result["open_set_reasons"]

    st.subheader("Binary RF output (malicious vs benign)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("p_benign", f"{rf_out.p_benign:.6f}")
    with col2:
        st.metric("p_malicious", f"{rf_out.p_malicious:.6f}")
    with col3:
        st.metric("RF confidence", f"{rf_out.rf_confidence:.6f}")

    st.subheader("Family RF output (malicious-only)")
    col4, col5, col6 = st.columns(3)

    # --- CHANGED HERE: Always show the actual prediction ---
    display_family_pred = str(family_out.family_pred)
    
    with col4:
        st.metric("family_pred", display_family_pred)
    with col5:
        st.metric("family_conf", f"{family_out.family_conf:.6f}")
    with col6:
        # If unknown, we can display this in red or just normally
        st.metric("family_unknown", str(family_out.family_unknown))

    # Add a small caption if unknown, rather than hiding the label
    if family_out.family_unknown:
        st.caption("⚠️ Family RF confidence is below threshold. This prediction may be inaccurate.")

    st.subheader("Open-set label (heuristic, model)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model open-set label", open_set_label)
    with c2:
        st.metric("Suggested Family (Logic)", derived_family)

    if reasons:
        st.markdown("**Reasons:**")
        for r in reasons:
            st.markdown(f"- {r}")


def render_rag_section(result: dict):
    rag = result["rag"]

    st.subheader("RAG neighbor summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("sim_max", f"{rag.sim_max:.6f}")
    with col2:
        st.metric("sim_mean", f"{rag.sim_mean:.6f}")
    with col3:
        st.metric("mal_frac", f"{rag.mal_frac:.3f}")
    with col4:
        st.metric("family_diversity", str(rag.family_diversity))

    st.markdown("**Top-k similar historical flows:**")
    render_neighbor_table(rag)


def render_llm_section(result: dict):
    st.subheader("LLM explanation + decision + action")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final decision (after policy)", result["final_decision"])
    with col2:
        st.metric("Recommended action (after policy)", result["final_action"])

    st.markdown("**Raw LLM output (including policy post-append):**")
    st.text_area(
        "LLM output",
        value=result["llm_text"],
        height=300,
    )


# -----------------------------
# Main Streamlit UI
# -----------------------------
def main():
    st.title("UNSW Open-set Net-RAG + LLM Demo")
    st.write(
        "Interactive wrapper around your `unsw_open_set_rag_demo.py` pipeline:\n"
        "- Binary RF for malicious vs benign\n"
        "- Family RF for attack family (known malicious)\n"
        "- Open-set labels: `KNOWN_BENIGN`, `KNOWN_MALICIOUS`, `UNKNOWN_MALICIOUS`\n"
        "- Net-RAG via Chroma over historical flows\n"
        "- LLM explanation + decision + action"
    )

    with st.spinner("Initializing models, Chroma, and loading data..."):
        pipeline = init_pipeline()

    df_test_known = pipeline["df_test_known"]
    df_test_unknown = pipeline["df_test_unknown"]

    st.sidebar.header("Flow selection")
    mode = st.sidebar.radio(
        "Selection mode",
        ["Random sample", "By index"],
    )

    result = None

    # ---------- RANDOM SAMPLE MODE ----------
    if mode == "Random sample":
        label_type = st.sidebar.selectbox(
            "Target *ground-truth* open-set type for random sample",
            ["KNOWN_BENIGN", "KNOWN_MALICIOUS", "UNKNOWN_MALICIOUS"],
        )

        error_only = False

        if st.sidebar.button("Sample and analyze"):
            row, df_name = sample_row_for_random(
                label_type,
                pipeline,
                df_test_known,
                df_test_unknown,
                error_only,
            )

            if row is None:
                st.error(f"No candidates found for {label_type}.")
            else:
                with st.spinner("Running full pipeline on sampled flow..."):
                    result = analyze_single_flow(row, df_name, pipeline)

    # ---------- BY INDEX MODE ----------
    else:  # By index
        df_choice = st.sidebar.selectbox(
            "Dataset",
            ["test_known", "test_unknown"],
        )
        index_mode = st.sidebar.radio(
            "Index mode",
            ["By position (iloc)", "By UNSW row index (df.index)"],
        )

        if df_choice == "test_known":
            df = df_test_known
        else:
            df = df_test_unknown

        st.sidebar.markdown(
            f"Current DataFrame: **{df_choice}** with **{len(df):,} rows**."
        )

        if index_mode == "By position (iloc)":
            pos = st.sidebar.number_input(
                "Position (0-based iloc)",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=0,
                step=1,
            )
            if st.sidebar.button("Analyze this position"):
                row = df.iloc[int(pos)]
                with st.spinner("Running full pipeline on selected flow..."):
                    result = analyze_single_flow(row, df_choice, pipeline)
        else:
            # By UNSW original row index (df.index)
            idx_min = int(df.index.min()) if len(df) > 0 else 0
            idx_max = int(df.index.max()) if len(df) > 0 else 0
            idx = st.sidebar.number_input(
                "UNSW row index (df.index)",
                min_value=idx_min,
                max_value=idx_max,
                value=idx_min,
                step=1,
            )
            if st.sidebar.button("Analyze this UNSW row index"):
                idx = int(idx)
                if idx not in df.index:
                    st.error(
                        f"Row index {idx} not present in {df_choice}. "
                        f"Valid indices range roughly from {idx_min} to {idx_max}, "
                        "but the index may be sparse due to filtering."
                    )
                else:
                    row = df.loc[idx]
                    with st.spinner("Running full pipeline on selected flow..."):
                        result = analyze_single_flow(row, df_choice, pipeline)

    # ---------- RENDER RESULT ----------
    if result is not None:
        st.markdown("---")
        st.header("Flow analysis")

        render_flow_summary(result)

        st.markdown("---")
        render_model_outputs(result)

        st.markdown("---")
        render_rag_section(result)

        st.markdown("---")
        render_llm_section(result)


if __name__ == "__main__":
    main()
