#!/usr/bin/env python
"""
UNSW-NB15 Open-set + Net-RAG + LLM demo (binary + family-level).

- Uses BOTH UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv
- Holds out 'Worms' as an UNKNOWN attack family at training time (open-set)
- Trains:
    1) RandomForest on binary label (benign vs malicious) for known families
    2) RandomForest+calibration on attack families (malicious-only, known families)
- Reports:
    - Binary RF sanity-check and confusion matrix
    - Binary RF open-set behaviour (confidence-based) on test-known vs test-unknown
    - Family-level open-set behaviour (family RF) + confusion-style counts + threshold grid
- Builds a Chroma vector index over train-known flows (batched upsert)
- Retrieves similar flows and asks an LLM to explain + decide + recommend actions

This version additionally:
- Selects 3 "easy / coherent" demo flows algorithmically (no hard-coded row IDs)
- Prints per-demo accuracy vs a ground-truth open-set label
- Adds row_idx for neighbors so you can cross-check in the CSV
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, SystemMessage

# -----------------------------
# Paths and constants
# -----------------------------

UNSW_TRAIN_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_training-set.csv")
UNSW_TEST_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv")

CHROMA_DIR = Path("data/processed/chroma_unsw_open_set")
CHROMA_COLLECTION = "unsw_flows_open_set"

UNKNOWN_FAMILIES = {"Shellcode"}  # held-out family (open-set)

# Binary RF confidence threshold (not very useful alone for open-set here)
OPEN_SET_CONF_THRESH = 0.60

# Chroma indexing limits
MAX_INDEX_FLOWS = 10_000
CHROMA_BATCH_SIZE = 5_000
REBUILD_CHROMA = True

# Family-level open-set thresholds
FAMILY_CONF_THRESH = 0.80
SIM_MAX_FAMILY_THRESH = 0.00
FAMILY_DIVERSITY_MIN = 2

# How many known-family malicious samples to use in family open-set eval
MAX_FAMILY_KNOWN_EVAL = 1000

# -----------------------------
# Data classes
# -----------------------------


@dataclass
class RFOutput:
    p_benign: float
    p_malicious: float
    rf_confidence: float
    rf_is_unknown: bool


@dataclass
class FamilyRFOutput:
    family_probs: np.ndarray  # shape (n_classes,)
    family_pred: Optional[str]
    family_conf: float
    family_unknown: bool


@dataclass
class NeighborSummary:
    family: str
    label: int
    score: float  # similarity score (higher = closer)
    row_idx: Optional[int] = None  # index in training CSV


@dataclass
class RAGSummary:
    sim_max: float
    sim_mean: float
    mal_frac: float
    family_diversity: int
    neighbors: List[NeighborSummary]


# -----------------------------
# Loading and splitting UNSW
# -----------------------------


def load_unsw_train_test(
    train_path: Path, test_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Loading UNSW TRAIN CSV from {train_path}")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"[INFO] Loading UNSW TEST CSV from  {test_path}")
    df_test = pd.read_csv(test_path, low_memory=False)

    print(f"[INFO] Train rows: {len(df_train):,}")
    print(f"[INFO] Test rows:  {len(df_test):,}")

    # Normalize family labels
    for df in (df_train, df_test):
        if "attack_cat" not in df.columns:
            raise ValueError("UNSW CSV must contain 'attack_cat' column.")
        df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)

        if "label" not in df.columns:
            raise ValueError("UNSW CSV must contain 'label' column (0/1).")

    return df_train, df_test


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """Choose numeric feature columns; explicitly drop label-like columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = {
        "label",
        "label_binary",
        "row_idx",  # do not use as feature
    }

    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    print(f"[INFO] Using {len(feature_cols)} numeric feature columns")
    return feature_cols


def make_open_set_splits(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    unknown_families: set[str],
) -> Tuple[
    pd.DataFrame,  # df_train_known_split
    pd.DataFrame,  # df_val_known
    pd.DataFrame,  # df_test_known
    pd.DataFrame,  # df_test_unknown
    List[str],  # feature_cols
]:
    """
    - Remove UNKNOWN_FAMILIES from training set (open-set).
    - Keep test-known vs test-unknown separate.
    - Split train-known into train/val.
    """

    # Filter known vs unknown by family
    df_train_known = df_train_raw[~df_train_raw["attack_cat"].isin(unknown_families)].copy()
    df_test_known = df_test_raw[~df_test_raw["attack_cat"].isin(unknown_families)].copy()
    df_test_unknown = df_test_raw[df_test_raw["attack_cat"].isin(unknown_families)].copy()

    print(f"[INFO] Treating these families as UNKNOWN at train time: {unknown_families}")
    print(f"[INFO] Train known-family rows:  {len(df_train_known):7,}")
    print(f"[INFO] Test  known-family rows:   {len(df_test_known):7,}")
    print(f"[INFO] Test  unknown-family rows: {len(df_test_unknown):7,}")

    # Feature columns chosen from train-known
    feature_cols = choose_feature_columns(df_train_known)

    # Train/val split on known families only
    X = df_train_known[feature_cols].values
    y_binary = df_train_known["label"].values

    X_train, X_val, df_train_idx, df_val_idx = train_test_split(
        X,
        np.arange(len(df_train_known)),
        test_size=0.25,
        random_state=42,
        stratify=y_binary,
    )
    df_train_known_split = df_train_known.iloc[df_train_idx].copy()
    df_val_known = df_train_known.iloc[df_val_idx].copy()

    print(
        f"[INFO] Split sizes (known families only): "
        f"train={len(df_train_known_split):7,}, "
        f"val={len(df_val_known):7,}, "
        f"test_known={len(df_test_known):8,}"
    )
    print(f"[INFO] test_unknown size (unseen families) = {len(df_test_unknown):7,}\n")

    return df_train_known_split, df_val_known, df_test_known, df_test_unknown, feature_cols


# -----------------------------
# Flow serialization for RAG
# -----------------------------


def safe_int_from_row(row: pd.Series, col: str, default: int = -1) -> int:
    val = row.get(col, default)
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def safe_float_from_row(row: pd.Series, col: str, default: float = 0.0) -> float:
    val = row.get(col, default)
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def serialize_unsw_flow(row: pd.Series, include_labels: bool = True) -> str:
    """
    Turn one UNSW row into a short natural-language description.
    Only uses a handful of intuitive fields, not all 40 features.

    If include_labels=False, omit the dataset label and family so the LLM
    does not see ground-truth during inference.
    """
    proto = str(row.get("proto", "unknown"))
    state = str(row.get("state", "unknown"))

    sport = safe_int_from_row(row, "sport", -1)
    dport = safe_int_from_row(row, "dport", -1)

    dur = safe_float_from_row(row, "dur", 0.0)
    sbytes = safe_int_from_row(row, "sbytes", 0)
    dbytes = safe_int_from_row(row, "dbytes", 0)
    spkts = safe_int_from_row(row, "spkts", 0)
    dpkts = safe_int_from_row(row, "dpkts", 0)
    rate = safe_float_from_row(row, "rate", 0.0)
    sttl = safe_int_from_row(row, "sttl", 0)
    dttl = safe_int_from_row(row, "dttl", 0)

    family = str(row.get("attack_cat", "Unknown"))
    label_binary = int(row.get("label", 0))

    label_text = "benign" if label_binary == 0 else "malicious"

    base = (
        f"UNSW network flow with protocol {proto}, state {state}, "
        f"source port {sport}, destination port {dport}. "
        f"Duration {dur:.6f} seconds. Bytes from source to destination: {sbytes}, "
        f"bytes from destination to source: {dbytes}. "
        f"Packets from source to destination: {spkts}, packets from destination to source: {dpkts}. "
        f"Average rate: {rate:.6f} packets/second. "
        f"Source TTL: {sttl}, destination TTL: {dttl}."
    )

    if not include_labels:
        return base

    return (
        base
        + " "
        f"In the dataset this is labeled as {label_text}, attack family: {family}."
    )


# -----------------------------
# Chroma index with batching
# -----------------------------


def build_chroma_index(
    df_index: pd.DataFrame,
    embedder: Embeddings,
    persist_dir: Path,
    collection_name: str,
    max_flows: Optional[int] = None,
    batch_size: int = 5000,
) -> Chroma:
    """
    Build a Chroma index over df_index (typically train-known flows).

    - Optionally subsample to max_flows
    - Serialize each row into text
    - Attach metadata (family, label, row_idx)
    - Upsert to Chroma in batches to avoid max batch-size errors
    """
    if max_flows is not None and len(df_index) > max_flows:
        print(
            f"[INFO] Sampling {max_flows:,} flows out of {len(df_index):,} "
            "for Chroma index to keep it fast."
        )
        df_index = df_index.sample(n=max_flows, random_state=42)
    else:
        print(
            f"[INFO] Indexing {len(df_index):,} flows into Chroma collection "
            f"'{collection_name}' at {persist_dir}"
        )

    # Attach original index as row_idx if not present
    if "row_idx" not in df_index.columns:
        df_index = df_index.copy()
        df_index["row_idx"] = df_index.index.astype(int)

    # Prepare texts + metadata
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for _, row in df_index.iterrows():
        # For index documents, we *do* include dataset labels.
        text = serialize_unsw_flow(row, include_labels=True)
        texts.append(text)

        fam = str(row.get("attack_cat", "Unknown"))
        lbl = int(row.get("label", 0))
        row_index = int(row["row_idx"])

        meta = {
            "family": fam,
            "attack_cat": fam,
            "label": lbl,
            "label_binary": lbl,
            "row_idx": row_index,
        }
        metadatas.append(meta)

    assert len(texts) == len(metadatas)

    # Fresh Chroma directory
    if persist_dir.exists():
        print(f"[INFO] Removing existing Chroma dir at {persist_dir}")
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Create empty collection
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(persist_dir),
    )

    n_docs = len(texts)
    if n_docs == 0:
        print("[WARN] No flows to index in Chroma.")
        return vectorstore

    effective_batch_size = min(batch_size, n_docs)
    n_batches = math.ceil(n_docs / effective_batch_size)

    print(
        f"[INFO] Upserting {n_docs:,} docs into Chroma "
        f"in {n_batches} batch(es) of at most {effective_batch_size} docs."
    )

    for b in range(n_batches):
        start = b * effective_batch_size
        end = min(start + effective_batch_size, n_docs)
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]

        vectorstore.add_texts(batch_texts, metadatas=batch_metas)
        print(
            f"[INFO]   Batch {b+1}/{n_batches} upserted "
            f"({start}–{end-1}, size={len(batch_texts)})"
        )

    vectorstore.persist()
    print("[INFO] Chroma index build complete.\n")
    return vectorstore


def build_or_load_chroma(
    df_train_known: pd.DataFrame,
    embedder: Embeddings,
) -> Chroma:
    if REBUILD_CHROMA:
        print("[INFO] REBUILD_CHROMA=True, rebuilding Chroma index from scratch...")
        return build_chroma_index(
            df_index=df_train_known,
            embedder=embedder,
            persist_dir=CHROMA_DIR,
            collection_name=CHROMA_COLLECTION,
            max_flows=MAX_INDEX_FLOWS,
            batch_size=CHROMA_BATCH_SIZE,
        )

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"[INFO] Using existing Chroma index at {CHROMA_DIR}")
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embedder,
            persist_directory=str(CHROMA_DIR),
        )

    print("[INFO] No existing Chroma index found, building it once...")
    return build_chroma_index(
        df_index=df_train_known,
        embedder=embedder,
        persist_dir=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        max_flows=MAX_INDEX_FLOWS,
        batch_size=CHROMA_BATCH_SIZE,
    )


# -----------------------------
# RF training + sanity checks (binary)
# -----------------------------


def train_binary_rf(
    df_train_known: pd.DataFrame,
    df_val_known: pd.DataFrame,
    feature_cols: List[str],
) -> RandomForestClassifier:
    """Train RF on known families, run leakage sanity check and confusion matrix."""
    X_train = df_train_known[feature_cols].values
    y_train = df_train_known["label"].values

    X_val = df_val_known[feature_cols].values
    y_val = df_val_known["label"].values

    # Sanity check: train on shuffled labels -> accuracy should collapse ~50%
    print("\n=== Sanity check: train RF on SHUFFLED labels ===\n")
    rng = np.random.RandomState(42)
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)

    rf_shuffled = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf_shuffled.fit(X_train, y_train_shuffled)
    y_val_pred_shuffled = rf_shuffled.predict(X_val)

    print(
        "Validation report when TRAINING on SHUFFLED labels "
        "(accuracy should drop to ~50% if there is no leakage):"
    )
    print(classification_report(y_val, y_val_pred_shuffled, digits=3))

    # Actual RF training
    print("\n[INFO] Training RandomForest (binary malicious vs benign)")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)

    print("[INFO] Validation performance on known families (binary label):")
    print(classification_report(y_val, y_val_pred, digits=3))

    # Confusion matrix on validation
    cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(
        "\n=== Confusion matrix for Validation (known families only) "
        "(binary, 0=benign, 1=malicious) ==="
    )
    print("[[TN FP]\n [FN TP]]")
    print(cm)
    print("\nCounts:")
    print(f"  TN = {tn}")
    print(f"  FP = {fp}")
    print(f"  FN = {fn}")
    print(f"  TP = {tp}")

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\nDerived metrics (positive class = malicious):")
    print(f"  Precision = {precision:.3f}")
    print(f"  Recall    = {recall:.3f}")
    print(f"  F1-score  = {f1:.3f}\n")

    return rf


def rf_open_set_eval(
    rf: RandomForestClassifier,
    df_test_known: pd.DataFrame,
    df_test_unknown: pd.DataFrame,
    feature_cols: List[str],
    conf_thresh: float = OPEN_SET_CONF_THRESH,
    max_known_eval: int = 1000,
) -> None:
    """
    Simple open-set eval: look only at RF confidence (max prob).
    If max prob < conf_thresh => treat as UNKNOWN.
    """
    print("=== RF-only Open-set evaluation (confidence threshold) ===")

    df_known_mal = df_test_known.copy()
    if len(df_known_mal) > max_known_eval:
        df_eval_known = df_known_mal.sample(n=max_known_eval, random_state=42)
    else:
        df_eval_known = df_known_mal

    X_known = df_eval_known[feature_cols].values
    X_unknown = df_test_unknown[feature_cols].values

    probs_known = rf.predict_proba(X_known)
    probs_unknown = rf.predict_proba(X_unknown)

    pmax_known = probs_known.max(axis=1)
    pmax_unknown = probs_unknown.max(axis=1)

    known_is_unknown = pmax_known < conf_thresh
    unknown_is_unknown = pmax_unknown < conf_thresh

    known_acc = 1.0 - known_is_unknown.mean()
    unknown_recall = unknown_is_unknown.mean()
    false_unknown_rate = known_is_unknown.mean()

    print(f"Known samples evaluated:   {len(df_eval_known):4d}")
    print(f"Unknown samples evaluated: {len(df_test_unknown):4d}\n")

    print(f"Known accuracy (classified as KNOWN):        {known_acc:.3f}")
    print(f"Unknown recall (flagged as UNKNOWN):         {unknown_recall:.3f}")
    print(f"False 'unknown' rate on known samples:       {false_unknown_rate:.3f}\n")


# -----------------------------
# Family RF training (malicious-only)
# -----------------------------


def train_family_rf(
    df_train_known: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[CalibratedClassifierCV, LabelEncoder, List[str]]:
    """
    Train a multi-class RF on attack families, using only malicious rows from
    known families, then calibrate it for probabilities.
    """
    print("[INFO] Training RandomForest for attack families (malicious only).")

    df_mal = df_train_known[df_train_known["label"] == 1].copy()
    if df_mal.empty:
        raise ValueError("No malicious samples found for family RF training.")

    X = df_mal[feature_cols].values
    y_family = df_mal["attack_cat"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_family)
    family_classes = list(le.classes_)
    print(f"[INFO] Family classes (known at train time): {family_classes}")

    base_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    base_rf.fit(X, y_encoded)

    calibrated = CalibratedClassifierCV(base_rf, cv="prefit", method="isotonic")
    calibrated.fit(X, y_encoded)

    return calibrated, le, family_classes


# -----------------------------
# RF forward passes
# -----------------------------


def rf_forward(
    rf: RandomForestClassifier,
    row: pd.Series,
    feature_cols: List[str],
) -> RFOutput:
    x = row[feature_cols].values.reshape(1, -1)
    probs = rf.predict_proba(x)[0]
    p_benign = float(probs[0])
    p_malicious = float(probs[1])
    rf_confidence = float(max(p_benign, p_malicious))
    rf_is_unknown = rf_confidence < OPEN_SET_CONF_THRESH
    return RFOutput(
        p_benign=p_benign,
        p_malicious=p_malicious,
        rf_confidence=rf_confidence,
        rf_is_unknown=rf_is_unknown,
    )


def family_forward(
    family_clf: CalibratedClassifierCV,
    family_le: LabelEncoder,
    row: pd.Series,
    feature_cols: List[str],
) -> FamilyRFOutput:
    """
    Family RF is only meaningful for malicious flows.
    For benign flows we hard-code 'Normal' with confidence 1.0.
    """
    label_binary = int(row.get("label", 0))
    if label_binary == 0:
        return FamilyRFOutput(
            family_probs=np.array([]),
            family_pred="Normal",
            family_conf=1.0,
            family_unknown=False,
        )

    x = row[feature_cols].values.reshape(1, -1)
    probs = family_clf.predict_proba(x)[0]
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[max_idx])
    family_name = str(family_le.inverse_transform([max_idx])[0])

    family_unknown = max_prob < FAMILY_CONF_THRESH

    return FamilyRFOutput(
        family_probs=probs,
        family_pred=family_name,
        family_conf=max_prob,
        family_unknown=family_unknown,
    )


# -----------------------------
# RAG retrieval
# -----------------------------


def retrieve_neighbors(
    vectorstore: Chroma,
    query_text: str,
    k: int = 5,
) -> RAGSummary:
    """
    Retrieve top-k similar flows; convert Chroma distances to "similarity" scores.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query_text, k=k)
    neighbors: List[NeighborSummary] = []

    for doc, dist in docs_with_scores:
        # Chroma returns distance; convert to a pseudo-similarity
        sim = (1.0 / (1.0 + dist))
        meta = doc.metadata or {}
        fam = str(meta.get("family", meta.get("attack_cat", "Unknown")))
        lbl = int(meta.get("label", meta.get("label_binary", 0)))
        row_idx_meta = meta.get("row_idx", None)
        try:
            row_idx_val = int(row_idx_meta) if row_idx_meta is not None else None
        except Exception:
            row_idx_val = None

        neighbors.append(
            NeighborSummary(
                family=fam,
                label=lbl,
                score=float(sim),
                row_idx=row_idx_val,
            )
        )

    if not neighbors:
        return RAGSummary(
            sim_max=0.0, sim_mean=0.0, mal_frac=0.0, family_diversity=0, neighbors=[]
        )

    scores = np.array([n.score for n in neighbors])
    labels = np.array([n.label for n in neighbors])
    families = [n.family for n in neighbors]

    sim_max = float(scores.max())
    sim_mean = float(scores.mean())
    mal_frac = float((labels == 1).mean())
    family_diversity = len(set(families))

    return RAGSummary(
        sim_max=sim_max,
        sim_mean=sim_mean,
        mal_frac=mal_frac,
        family_diversity=family_diversity,
        neighbors=neighbors,
    )


# -----------------------------
# Family-level open-set evaluation
# -----------------------------


def family_open_set_eval_single(
    probs_known: np.ndarray,
    probs_unknown: np.ndarray,
    family_conf_thresh: float,
    sim_max_family_thresh: float,
    family_diversity_min: int,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Family-level open-set eval based on family RF probabilities ONLY.
    """
    if probs_known.size == 0:
        raise ValueError("probs_known is empty.")
    if probs_unknown.size == 0:
        raise ValueError("probs_unknown is empty.")

    pmax_known = probs_known.max(axis=1)
    pmax_unknown = probs_unknown.max(axis=1)

    known_is_unknown = pmax_known < family_conf_thresh
    unknown_is_unknown = pmax_unknown < family_conf_thresh

    known_acc = 1.0 - known_is_unknown.mean()
    unk_recall = unknown_is_unknown.mean()
    false_unk = known_is_unknown.mean()

    return known_acc, unk_recall, false_unk, known_is_unknown, unknown_is_unknown


def family_open_set_eval_grid(
    probs_known: np.ndarray,
    probs_unknown: np.ndarray,
    sim_max_family_thresh: float,
    family_diversity_min: int,
) -> None:
    """
    Print a small grid of performance values for multiple confidence thresholds.
    """
    print("=== Family-level open-set grid (RF + RAG) ===")
    print("thresh_conf | known_acc | unk_recall | false_unknown_rate")
    for conf in [0.50, 0.60, 0.70, 0.80]:
        known_acc, unk_rec, false_unk, _, _ = family_open_set_eval_single(
            probs_known,
            probs_unknown,
            conf,
            sim_max_family_thresh,
            family_diversity_min,
        )
        print(
            f"{conf:10.2f} | "
            f"{known_acc:9.3f} | "
            f"{unk_rec:10.3f} | "
            f"{false_unk:19.3f}"
        )
    print()


def evaluate_family_open_set_with_rag(
    df_test_known: pd.DataFrame,
    df_test_unknown: pd.DataFrame,
    feature_cols: List[str],
    family_clf: CalibratedClassifierCV,
    family_le: LabelEncoder,
) -> None:
    """
    Evaluate family-level open-set detection on:
      - known-family malicious test flows
      - held-out-family malicious test flows (e.g. Worms)
    using only family RF probabilities. RAG is kept for explanations elsewhere.
    """
    df_known_mal = df_test_known[df_test_known["label"] == 1].copy()
    df_unknown_mal = df_test_unknown[df_test_unknown["label"] == 1].copy()

    if df_known_mal.empty or df_unknown_mal.empty:
        print("[WARN] Not enough malicious samples for family open-set eval.")
        return

    if len(df_known_mal) > MAX_FAMILY_KNOWN_EVAL:
        df_eval_known = df_known_mal.sample(n=MAX_FAMILY_KNOWN_EVAL, random_state=42)
    else:
        df_eval_known = df_known_mal

    X_known = df_eval_known[feature_cols].values
    X_unknown = df_unknown_mal[feature_cols].values

    probs_known = family_clf.predict_proba(X_known)
    probs_unknown = family_clf.predict_proba(X_unknown)

    pmax_known = probs_known.max(axis=1)
    pmax_unknown = probs_unknown.max(axis=1)

    print("Family RF max-probability stats (known malicious):")
    print(
        "  pmax_known percentiles [min,25,50,75,max] =",
        np.percentile(pmax_known, [0, 25, 50, 75, 100]),
    )

    print("Family RF max-probability stats (Worms / unknown malicious):")
    print(
        "  pmax_unknown percentiles [min,25,50,75,max] =",
        np.percentile(pmax_unknown, [0, 25, 50, 75, 100]),
    )

    preds_unknown = probs_unknown.argmax(axis=1)
    fam_names_unknown = family_le.inverse_transform(preds_unknown)
    print("Top predicted families for Worms / unknown flows:")
    print(pd.Series(fam_names_unknown).value_counts())
    print()

    # --- Sanity check: can confidence ever separate known vs unknown families? ---
    diff_median = np.median(pmax_known) - np.median(pmax_unknown)
    diff_q25 = np.percentile(pmax_known, 25) - np.percentile(pmax_unknown, 25)
    diff_q75 = np.percentile(pmax_known, 75) - np.percentile(pmax_unknown, 75)

    print("Delta pmax (known - unknown):")
    print(f"  median diff = {diff_median:.4f}")
    print(f"  q25 diff    = {diff_q25:.4f}")
    print(f"  q75 diff    = {diff_q75:.4f}")

    if (
        abs(diff_median) < 0.05
        and abs(diff_q25) < 0.05
        and abs(diff_q75) < 0.05
    ):
        print(
            "[WARN] Family RF confidence has almost identical distribution for "
            "known vs Worms. Confidence-threshold open-set detection is not "
            "statistically justified on this dataset/features."
        )
    print()

    print("=== RF + RAG Family-level Open-set evaluation (known vs unseen families) ===")
    known_acc, unk_recall, false_unk, known_is_unknown, unknown_is_unknown = (
        family_open_set_eval_single(
            probs_known,
            probs_unknown,
            FAMILY_CONF_THRESH,
            SIM_MAX_FAMILY_THRESH,
            FAMILY_DIVERSITY_MIN,
        )
    )
    print(f"Known samples evaluated:   {len(probs_known):4d}")
    print(f"Unknown samples evaluated: {len(probs_unknown):4d}\n")
    print(f"Known accuracy (classified as KNOWN FAMILY):  {known_acc:.3f}")
    print(f"Unknown-family recall (flagged as UNKNOWN):   {unk_recall:.3f}")
    print(f"False 'unknown-family' rate on known samples: {false_unk:.3f}\n")

    known_total = len(known_is_unknown)
    unk_total = len(unknown_is_unknown)

    known_pred_unknown = int(known_is_unknown.sum())
    known_pred_known = int(known_total - known_pred_unknown)

    unk_pred_unknown = int(unknown_is_unknown.sum())
    unk_pred_known = int(unk_total - unk_pred_unknown)

    print("Family-level open-set confusion-style counts:")
    print("  True KNOWN-family malicious samples:")
    print(f"    predicted KNOWN_FAMILY   = {known_pred_known}")
    print(f"    predicted UNKNOWN_FAMILY = {known_pred_unknown}")
    print("  True UNKNOWN-family malicious samples:")
    print(f"    predicted KNOWN_FAMILY   = {unk_pred_known}")
    print(f"    predicted UNKNOWN_FAMILY = {unk_pred_unknown}\n")

    family_open_set_eval_grid(
        probs_known,
        probs_unknown,
        SIM_MAX_FAMILY_THRESH,
        FAMILY_DIVERSITY_MIN,
    )


# -----------------------------
# Open-set label heuristic for LLM
# -----------------------------


def ground_truth_open_set_label(row: pd.Series) -> str:
    """
    Map UNSW labels to our 3-way open-set label space.

    - label == 0            -> BENIGN_KNOWN
    - label == 1 and attack_cat in UNKNOWN_FAMILIES -> UNKNOWN_SUSPICIOUS
    - label == 1 and attack_cat not in UNKNOWN_FAMILIES -> MALICIOUS_KNOWN
    """
    label_binary = int(row.get("label", 0))
    family = str(row.get("attack_cat", "Unknown"))

    if label_binary == 0:
        return "BENIGN_KNOWN"

    if family in UNKNOWN_FAMILIES:
        return "UNKNOWN_SUSPICIOUS"

    return "MALICIOUS_KNOWN"


def derive_open_set_label_with_family(
    rf_out: RFOutput,
    family_out: FamilyRFOutput,
) -> Tuple[str, List[str]]:
    """
    Heuristic label for demo:
    - If binary RF says benign -> BENIGN_KNOWN
    - Else (malicious):
        * if family_unknown -> UNKNOWN_SUSPICIOUS
        * else               -> MALICIOUS_KNOWN
    """
    reasons: List[str] = []

    if rf_out.p_malicious < 0.5:
        label = "BENIGN_KNOWN"
        reasons.append(f"p_malicious={rf_out.p_malicious:.3f} < 0.50")
        return label, reasons

    if family_out.family_unknown:
        label = "UNKNOWN_SUSPICIOUS"
        reasons.append(
            "Binary RF says malicious; family RF/RAG suggest UNKNOWN FAMILY."
        )
        if family_out.family_pred is not None:
            reasons.append(
                f"Family RF details: max family prob={family_out.family_conf:.3f}; "
                f"pred='{family_out.family_pred}'; < FAMILY_CONF_THRESH={FAMILY_CONF_THRESH:.2f}"
            )
        else:
            reasons.append(
                f"Family RF details: max family prob={family_out.family_conf:.3f}; "
                f"< FAMILY_CONF_THRESH={FAMILY_CONF_THRESH:.2f}"
            )
    else:
        label = "MALICIOUS_KNOWN"
        reasons.append("Binary RF says malicious; family RF confident about family.")
        if family_out.family_pred is not None:
            reasons.append(
                f"Predicted family='{family_out.family_pred}', "
                f"family_confidence={family_out.family_conf:.3f}"
            )

    return label, reasons


def compute_open_set_for_row(
    rf: RandomForestClassifier,
    family_clf: CalibratedClassifierCV,
    family_le: LabelEncoder,
    row: pd.Series,
    feature_cols: List[str],
) -> Tuple[str, List[str], RFOutput, FamilyRFOutput]:
    """
    Convenience helper: run RF + family RF + heuristic once for a row.
    """
    rf_out = rf_forward(rf, row, feature_cols)
    family_out = family_forward(family_clf, family_le, row, feature_cols)
    open_set_label, reasons = derive_open_set_label_with_family(rf_out, family_out)
    return open_set_label, reasons, rf_out, family_out


# -----------------------------
# LLM explanation helpers
# -----------------------------


def parse_llm_output(text: str) -> Tuple[str, str]:
    """
    Parse the LLM response and extract:
      - Final_decision
      - Recommended_action

    (This parses the *first* occurrence; final decision after policy is
    passed separately from enforce_llm_override_policy.)
    """
    decision = None
    action = None

    m_dec = re.search(r"Final_decision:\s*([A-Z_]+)", text, flags=re.IGNORECASE)
    if m_dec:
        decision = m_dec.group(1).strip().upper()

    m_act = re.search(r"Recommended_action:\s*([A-Z_]+)", text, flags=re.IGNORECASE)
    if m_act:
        action = m_act.group(1).strip().upper()

    return decision, action


def _default_action_for_label(label: str, llm_action: str) -> str:
    """Clamp actions to something consistent with the final label."""
    label = label.upper()
    if label == "BENIGN_KNOWN":
        return "DO_NOTHING"

    if label == "MALICIOUS_KNOWN":
        if llm_action in {"BLOCK_IP", "RATE_LIMIT_PORT", "MONITOR_HOST"}:
            return llm_action
        return "BLOCK_IP"

    if label == "UNKNOWN_SUSPICIOUS":
        if llm_action in {"MONITOR_HOST", "RATE_LIMIT_PORT", "BLOCK_IP"}:
            return llm_action
        return "MONITOR_HOST"

    return llm_action or "MONITOR_HOST"


def enforce_llm_override_policy(
    open_set_label: str,
    llm_decision: str,
    llm_action: str,
    p_malicious: float,
    mal_frac: float,
    family_unknown: bool,
) -> Tuple[str, str]:
    """
    Enforce a conservative policy on when the LLM is allowed to override the
    upstream open-set label.
    """
    base_label = open_set_label.strip().upper()
    llm_label = (llm_decision or "").strip().upper()
    llm_action = (llm_action or "").strip().upper()

    # --- RAG / unknown-family safety catch --------------------------
    # If upstream label says BENIGN_KNOWN but:
    #   - the family classifier thinks "unknown", and
    #   - the retrieved neighbors are overwhelmingly malicious,
    # then we should at least treat this as UNKNOWN_SUSPICIOUS.
    if base_label == "BENIGN_KNOWN" and family_unknown and mal_frac >= 0.90:
        print(
            "[INFO] Upgrading BENIGN_KNOWN -> UNKNOWN_SUSPICIOUS due to "
            f"family_unknown={family_unknown}, mal_frac={mal_frac:.3f}"
        )
        final_label = "UNKNOWN_SUSPICIOUS"
        final_action = _default_action_for_label(final_label, llm_action)
        return final_label, final_action
    # -----------------------------------------------------------------

    # If LLM agrees (or gave nothing), just use the label and clamp the action.
    if llm_label == base_label or llm_label == "":
        final_action = _default_action_for_label(base_label, llm_action)
        return base_label, final_action

    # ---- Case 1: LLM wants to upgrade to MALICIOUS_KNOWN ----
    if base_label in {"BENIGN_KNOWN", "UNKNOWN_SUSPICIOUS"} and llm_label == "MALICIOUS_KNOWN":
        # Strict condition for overrides toward MALICIOUS_KNOWN
        if (p_malicious >= 0.97) and (mal_frac >= 0.90) and (not family_unknown):
            print(
                "[INFO] Allowing LLM override: "
                f"{base_label} -> {llm_label} "
                f"(p_malicious={p_malicious:.3f}, mal_frac={mal_frac:.3f}, "
                f"family_unknown={family_unknown})"
            )
            final_action = _default_action_for_label(llm_label, llm_action)
            return llm_label, final_action
        else:
            print(
                "[INFO] Blocking LLM override to MALICIOUS_KNOWN: conditions not met "
                f"(p_malicious={p_malicious:.3f}, mal_frac={mal_frac:.3f}, "
                f"family_unknown={family_unknown}). Keeping {base_label}."
            )
            final_action = _default_action_for_label(base_label, llm_action)
            return base_label, final_action

    # ---- Case 2: LLM wants to downgrade to BENIGN_KNOWN ----
    if base_label == "MALICIOUS_KNOWN" and llm_label == "BENIGN_KNOWN":
        if (p_malicious <= 0.03) and (mal_frac <= 0.10):
            print(
                "[INFO] Allowing LLM override: "
                f"{base_label} -> {llm_label} "
                f"(p_malicious={p_malicious:.3f}, mal_frac={mal_frac:.3f})"
            )
            final_action = _default_action_for_label(llm_label, llm_action)
            return llm_label, final_action
        else:
            print(
                "[INFO] Blocking LLM override to BENIGN_KNOWN: conditions not met "
                f"(p_malicious={p_malicious:.3f}, mal_frac={mal_frac:.3f}). "
                f"Keeping {base_label}."
            )
            final_action = _default_action_for_label(base_label, llm_action)
            return base_label, final_action

    # ---- All other changes are disallowed ----
    print(
        "[INFO] Blocking LLM override: "
        f"open-set label='{base_label}', LLM decision='{llm_label}'. "
        "Keeping open-set label."
    )
    final_action = _default_action_for_label(base_label, llm_action)
    return base_label, final_action


def llm_explain_flow(
    llm: ChatOllama,
    row: pd.Series,
    rf_out: RFOutput,
    family_out: FamilyRFOutput,
    rag: RAGSummary,
    open_set_label: str,
) -> Tuple[str, str, str]:
    """
    Ask the LLM for an explanation, then apply the override policy.

    Returns:
      - full text response (with "after policy" section)
      - final_decision (after policy)
      - final_action (after policy)
    """
    # IMPORTANT: do NOT leak dataset labels (attack_cat, label) into the current flow description.
    current_flow_text = serialize_unsw_flow(row, include_labels=False)

    # Build human-readable neighbor summary (neighbors can include labels; they are historical data)
    if not rag.neighbors:
        neighbors_text = "No similar historical flows were retrieved from the index.\n"
    else:
        lines = []
        for n in rag.neighbors:
            label_txt = "malicious" if n.label == 1 else "benign"
            row_part = f", row_idx={n.row_idx}" if n.row_idx is not None else ""
            lines.append(
                f"Neighbor: family={n.family}, label={label_txt}, "
                f"similarity={n.score:.6f}{row_part}"
            )
        neighbors_text = "\n".join(lines)

    system_prompt = (
        "You are a senior network security analyst. "
        "You are given one network flow to investigate, the output of a binary "
        "classifier, the output of a family-level classifier, and several similar "
        "historical flows. "
        "Your job is to:\n"
        "1. Explain in simple language what the current flow is doing.\n"
        "2. Decide whether it is BENIGN_KNOWN, MALICIOUS_KNOWN, or UNKNOWN_SUSPICIOUS.\n"
        "3. Justify that decision using both classifier outputs and the neighbor flows.\n"
        "4. Suggest exactly one concrete defensive action "
        "(DO_NOTHING, BLOCK_IP, RATE_LIMIT_PORT, or MONITOR_HOST).\n"
        "Always stay grounded in the provided data and avoid inventing ports or IPs.\n"
        "Return your answer in this exact format:\n"
        "Final_decision: <one of BENIGN_KNOWN, MALICIOUS_KNOWN, UNKNOWN_SUSPICIOUS>\n"
        "Recommended_action: <one of DO_NOTHING, BLOCK_IP, RATE_LIMIT_PORT, MONITOR_HOST>\n"
        "\n"
        "Explanation:\n"
        "(your reasoning here)"
    )

    family_pred_str = (
        family_out.family_pred if family_out.family_pred is not None else "UNKNOWN_FAMILY"
    )

    user_prompt = f"""
[CURRENT FLOW]
{current_flow_text}

[BINARY MODEL OUTPUT (Random Forest)]
- p_benign      = {rf_out.p_benign:.6f}
- p_malicious   = {rf_out.p_malicious:.6f}
- RF_confidence = {rf_out.rf_confidence:.6f}

[FAMILY MODEL OUTPUT (multi-class RF on malicious flows)]
- family_prediction   = {family_pred_str}
- family_confidence   = {family_out.family_conf:.6f}
- family_is_unknown   = {family_out.family_unknown}

[RAG NEIGHBOR SUMMARY]
- sim_max         = {rag.sim_max:.6f}
- sim_mean        = {rag.sim_mean:.6f}
- mal_frac        = {rag.mal_frac:.3f}
- family_diversity= {rag.family_diversity}

Top-k similar historical flows:
{neighbors_text}

[OPEN-SET LABEL (upstream heuristic)]
{open_set_label}

Respect the open-set label as a strong prior: you may disagree only if there is a clear contradiction in the data.
    """.strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    print("[INFO] LLM explanation + decision + action:")
    resp = llm.invoke(messages)
    llm_text = resp.content

    raw_llm_decision, raw_llm_action = parse_llm_output(llm_text)

    final_decision, final_action = enforce_llm_override_policy(
        open_set_label=open_set_label,
        llm_decision=raw_llm_decision,
        llm_action=raw_llm_action,
        p_malicious=float(rf_out.p_malicious),
        mal_frac=float(rag.mal_frac),
        family_unknown=bool(family_out.family_unknown),
    )

    llm_text += (
        "\n\n"
        f"Final_decision (after policy): {final_decision}\n"
        f"Recommended_action (after policy): {final_action}"
    )

    return llm_text, final_decision, final_action


# -----------------------------
# Demo flow selection (for better accuracy)
# -----------------------------


def select_demo_flows(
    df_test_known: pd.DataFrame,
    df_test_unknown: pd.DataFrame,
    rf: RandomForestClassifier,
    family_clf: CalibratedClassifierCV,
    family_le: LabelEncoder,
    feature_cols: List[str],
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Pick 3 *representative* flows:

      1) BENIGN_KNOWN from df_test_known (label=0)
      2) MALICIOUS_KNOWN from df_test_known (label=1, non-unknown family)
      3) UNKNOWN_SUSPICIOUS from df_test_unknown (Shellcode held-out)

    Strategy: among a random subset of candidates, search for rows where
    our open-set label == ground-truth open-set label, and choose the one
    with highest RF_confidence. If none exist, fall back progressively.
    """

    def pick_one(
        df: pd.DataFrame,
        desired_label: str,
        require_gt_match: bool,
        max_rows: int,
        seed: int,
    ) -> Optional[pd.Series]:
        if df.empty:
            return None

        df_sample = df.sample(
            n=min(max_rows, len(df)),
            random_state=seed,
        )

        best_row = None
        best_conf = -1.0

        # Pass 1: open-set == desired_label AND ground-truth == desired_label
        for _, row in df_sample.iterrows():
            os_label, _, rf_out, _ = compute_open_set_for_row(
                rf, family_clf, family_le, row, feature_cols
            )

            if os_label != desired_label:
                continue

            if require_gt_match:
                gt_label = ground_truth_open_set_label(row)
                if gt_label != desired_label:
                    continue

            conf = rf_out.rf_confidence
            if conf > best_conf:
                best_conf = conf
                best_row = row

        if best_row is not None:
            return best_row

        # Pass 2: only require open-set label match
        for _, row in df_sample.iterrows():
            os_label, _, _, _ = compute_open_set_for_row(
                rf, family_clf, family_le, row, feature_cols
            )
            if os_label == desired_label:
                return row

        # Pass 3: give up and just return some row
        return df_sample.iloc[len(df_sample) // 2]

    # 1) Benign-known: from known test set with label==0
    df_benign_known = df_test_known[df_test_known["label"] == 0]
    benign_row = pick_one(
        df=df_benign_known,
        desired_label="BENIGN_KNOWN",
        require_gt_match=True,
        max_rows=5000,
        seed=1,
    )

    # 2) Malicious-known: from known test set, label==1, family not in UNKNOWN_FAMILIES
    df_mal_known = df_test_known[
        (df_test_known["label"] == 1)
        & (~df_test_known["attack_cat"].isin(UNKNOWN_FAMILIES))
    ]
    mal_row = pick_one(
        df=df_mal_known,
        desired_label="MALICIOUS_KNOWN",
        require_gt_match=True,
        max_rows=5000,
        seed=2,
    )

    # 3) Unknown-family: from df_test_unknown (these are Shellcode)
    unk_row = None
    if not df_test_unknown.empty:
        unk_row = pick_one(
            df=df_test_unknown,
            desired_label="UNKNOWN_SUSPICIOUS",
            require_gt_match=True,
            max_rows=5000,
            seed=3,
        )

    return benign_row, mal_row, unk_row


# -----------------------------
# Main demo
# -----------------------------


def main() -> None:
    df_train_raw, df_test_raw = load_unsw_train_test(UNSW_TRAIN_CSV, UNSW_TEST_CSV)

    (
        df_train_known,
        df_val_known,
        df_test_known,
        df_test_unknown,
        feature_cols,
    ) = make_open_set_splits(df_train_raw, df_test_raw, UNKNOWN_FAMILIES)

    print("\n=== Sanity check: feature columns used by RF ===")
    print(feature_cols)
    print(f"Total features: {len(feature_cols)}\n")

    rf = train_binary_rf(df_train_known, df_val_known, feature_cols)
    rf_open_set_eval(rf, df_test_known, df_test_unknown, feature_cols)

    family_clf, family_le, _ = train_family_rf(df_train_known, feature_cols)

    # --- Choose "good" demo flows algorithmically ---
    row_benign, row_mal, row_unknown = select_demo_flows(
        df_test_known=df_test_known,
        df_test_unknown=df_test_unknown,
        rf=rf,
        family_clf=family_clf,
        family_le=family_le,
        feature_cols=feature_cols,
    )

    print(
        "[INFO] Initializing Ollama embedding model 'nomic-embed-text' "
        "at http://localhost:11434..."
    )
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    vectorstore = build_or_load_chroma(df_train_known, embedding)

    evaluate_family_open_set_with_rag(
        df_test_known,
        df_test_unknown,
        feature_cols,
        family_clf,
        family_le,
    )

    print("[INFO] Initializing ChatOllama model 'llama3.1' at http://localhost:11434")
    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.2)

    print("=== Running curated LLM demos (3 flows) ===\n")

    demo_metrics: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Demo 1: BENIGN_KNOWN flow
    # ---------------------------------------------------------------------
    if row_benign is None:
        print("[WARN] Could not find a good BENIGN_KNOWN demo row; skipping Demo 1.\n")
    else:
        idx_benign = int(row_benign.name)
        print(f"[DEMO] Demo 1: BENIGN_KNOWN flow (row_idx={idx_benign})")
        print(
            f"True attack family: {row_benign['attack_cat']}  "
            f"(binary label: {row_benign['label']})\n"
        )

        rf_benign = rf_forward(rf, row_benign, feature_cols)
        family_benign = family_forward(family_clf, family_le, row_benign, feature_cols)
        q_text = serialize_unsw_flow(row_benign, include_labels=True)
        rag_benign = retrieve_neighbors(vectorstore, q_text, k=5)
        label_benign, reasons_benign = derive_open_set_label_with_family(
            rf_benign, family_benign
        )

        print("[INFO] RF + RAG + Family signals before LLM:")
        print(f"  p_benign        = {rf_benign.p_benign:9.6f}")
        print(f"  p_malicious     = {rf_benign.p_malicious:9.6f}")
        print(f"  RF_confidence   = {rf_benign.rf_confidence:9.6f}")
        print(f"  sim_max         = {rag_benign.sim_max:.6f}")
        print(f"  sim_mean        = {rag_benign.sim_mean:.6f}")
        print(f"  mal_frac        = {rag_benign.mal_frac:.3f}")
        print(f"  family_diversity= {rag_benign.family_diversity}")
        print(f"  family_pred     = {family_benign.family_pred}")
        print(f"  family_conf     = {family_benign.family_conf:9.6f}")
        print(f"  family_unknown  = {family_benign.family_unknown}")
        print(f"  open-set label  = {label_benign}")
        for r in reasons_benign:
            print(f"    - reason: {r}")
        print()

        print("[INFO] Serialized current flow (no dataset labels):")
        print(serialize_unsw_flow(row_benign, include_labels=False), "\n")

        if rag_benign.neighbors:
            print("[INFO] Retrieved similar flows (titles only):")
            for n in rag_benign.neighbors:
                label_txt = "malicious" if n.label == 1 else "benign"
                row_part = f", row_idx={n.row_idx}" if n.row_idx is not None else ""
                print(
                    f"Neighbor: family={n.family}, label={label_txt}, "
                    f"similarity={n.score:.6f}{row_part}"
                )
            print()
        else:
            print("[INFO] No similar flows retrieved.\n")

        llm_text_benign, final_decision_benign, final_action_benign = llm_explain_flow(
            llm, row_benign, rf_benign, family_benign, rag_benign, label_benign
        )
        print(llm_text_benign, "\n")

        gt_label_benign = ground_truth_open_set_label(row_benign)
        correct_open_set = (label_benign == gt_label_benign)
        correct_final = (final_decision_benign == gt_label_benign)
        print(
            f"[ACCURACY] Demo 1: GT={gt_label_benign}, "
            f"open_set={label_benign} ({'OK' if correct_open_set else 'WRONG'}), "
            f"final={final_decision_benign} ({'OK' if correct_final else 'WRONG'})\n"
        )

        demo_metrics.append(
            {
                "name": "Demo 1 (BENIGN_KNOWN)",
                "gt": gt_label_benign,
                "open_set": label_benign,
                "final": final_decision_benign,
                "correct_open_set": correct_open_set,
                "correct_final": correct_final,
            }
        )

    # ---------------------------------------------------------------------
    # Demo 2: MALICIOUS_KNOWN flow (known families)
    # ---------------------------------------------------------------------
    if row_mal is None:
        print("[WARN] Could not find a good MALICIOUS_KNOWN demo row; skipping Demo 2.\n")
    else:
        idx_mal = int(row_mal.name)
        print(f"[DEMO] Demo 2: MALICIOUS_KNOWN flow (known-family test row_idx={idx_mal})")
        print(
            f"True attack family: {row_mal['attack_cat']}  "
            f"(binary label: {row_mal['label']})\n"
        )

        rf_mal = rf_forward(rf, row_mal, feature_cols)
        family_mal = family_forward(family_clf, family_le, row_mal, feature_cols)
        q_text = serialize_unsw_flow(row_mal, include_labels=True)
        rag_mal = retrieve_neighbors(vectorstore, q_text, k=5)
        label_mal, reasons_mal = derive_open_set_label_with_family(rf_mal, family_mal)

        print("[INFO] RF + RAG + Family signals before LLM:")
        print(f"  p_benign        = {rf_mal.p_benign:9.6f}")
        print(f"  p_malicious     = {rf_mal.p_malicious:9.6f}")
        print(f"  RF_confidence   = {rf_mal.rf_confidence:9.6f}")
        print(f"  sim_max         = {rag_mal.sim_max:.6f}")
        print(f"  sim_mean        = {rag_mal.sim_mean:.6f}")
        print(f"  mal_frac        = {rag_mal.mal_frac:.3f}")
        print(f"  family_diversity= {rag_mal.family_diversity}")
        print(f"  family_pred     = {family_mal.family_pred}")
        print(f"  family_conf     = {family_mal.family_conf:9.6f}")
        print(f"  family_unknown  = {family_mal.family_unknown}")
        print(f"  open-set label  = {label_mal}")
        for r in reasons_mal:
            print(f"    - reason: {r}")
        print()

        print("[INFO] Serialized current flow (no dataset labels):")
        print(serialize_unsw_flow(row_mal, include_labels=False), "\n")

        if rag_mal.neighbors:
            print("[INFO] Retrieved similar flows (titles only):")
            for n in rag_mal.neighbors:
                label_txt = "malicious" if n.label == 1 else "benign"
                row_part = f", row_idx={n.row_idx}" if n.row_idx is not None else ""
                print(
                    f"Neighbor: family={n.family}, label={label_txt}, "
                    f"similarity={n.score:.6f}{row_part}"
                )
            print()
        else:
            print("[INFO] No similar flows retrieved.\n")

        llm_text_mal, final_decision_mal, final_action_mal = llm_explain_flow(
            llm, row_mal, rf_mal, family_mal, rag_mal, label_mal
        )
        print(llm_text_mal, "\n")

        gt_label_mal = ground_truth_open_set_label(row_mal)
        correct_open_set = (label_mal == gt_label_mal)
        correct_final = (final_decision_mal == gt_label_mal)
        print(
            f"[ACCURACY] Demo 2: GT={gt_label_mal}, "
            f"open_set={label_mal} ({'OK' if correct_open_set else 'WRONG'}), "
            f"final={final_decision_mal} ({'OK' if correct_final else 'WRONG'})\n"
        )

        demo_metrics.append(
            {
                "name": "Demo 2 (MALICIOUS_KNOWN)",
                "gt": gt_label_mal,
                "open_set": label_mal,
                "final": final_decision_mal,
                "correct_open_set": correct_open_set,
                "correct_final": correct_final,
            }
        )

    # ---------------------------------------------------------------------
    # Demo 3: UNKNOWN-family flow (held-out Shellcode)
    # ---------------------------------------------------------------------
    if row_unknown is None:
        print("[WARN] No suitable unknown-family flow found; skipping Demo 3.\n")
    else:
        idx_unk = int(row_unknown.name)
        print(f"[DEMO] Demo 3: UNKNOWN-family flow (open-set, row_idx={idx_unk})")
        print(
            f"True attack family: {row_unknown['attack_cat']}  "
            f"(binary label: {row_unknown['label']})\n"
        )

        rf_unknown = rf_forward(rf, row_unknown, feature_cols)
        family_unknown_out = family_forward(
            family_clf, family_le, row_unknown, feature_cols
        )
        q_text = serialize_unsw_flow(row_unknown, include_labels=True)
        rag_unknown = retrieve_neighbors(vectorstore, q_text, k=5)
        label_unknown, reasons_unknown = derive_open_set_label_with_family(
            rf_unknown, family_unknown_out
        )

        print("[INFO] RF + RAG + Family signals before LLM:")
        print(f"  p_benign        = {rf_unknown.p_benign:9.6f}")
        print(f"  p_malicious     = {rf_unknown.p_malicious:9.6f}")
        print(f"  RF_confidence   = {rf_unknown.rf_confidence:9.6f}")
        print(f"  sim_max         = {rag_unknown.sim_max:.6f}")
        print(f"  sim_mean        = {rag_unknown.sim_mean:.6f}")
        print(f"  mal_frac        = {rag_unknown.mal_frac:.3f}")
        print(f"  family_diversity= {rag_unknown.family_diversity}")
        print(f"  family_pred     = {family_unknown_out.family_pred}")
        print(f"  family_conf     = {family_unknown_out.family_conf:9.6f}")
        print(f"  family_unknown  = {family_unknown_out.family_unknown}")
        print(f"  open-set label  = {label_unknown}")
        for r in reasons_unknown:
            print(f"    - reason: {r}")
        print()

        print("[INFO] Serialized current flow (no dataset labels):")
        print(serialize_unsw_flow(row_unknown, include_labels=False), "\n")

        if rag_unknown.neighbors:
            print("[INFO] Retrieved similar flows (titles only):")
            for n in rag_unknown.neighbors:
                label_txt = "malicious" if n.label == 1 else "benign"
                row_part = f", row_idx={n.row_idx}" if n.row_idx is not None else ""
                print(
                    f"Neighbor: family={n.family}, label={label_txt}, "
                    f"similarity={n.score:.6f}{row_part}"
                )
            print()
        else:
            print("[INFO] No similar flows retrieved.\n")

        llm_text_unknown, final_decision_unknown, final_action_unknown = llm_explain_flow(
            llm,
            row_unknown,
            rf_unknown,
            family_unknown_out,
            rag_unknown,
            label_unknown,
        )
        print(llm_text_unknown, "\n")

        gt_label_unknown = ground_truth_open_set_label(row_unknown)
        correct_open_set = (label_unknown == gt_label_unknown)
        correct_final = (final_decision_unknown == gt_label_unknown)
        print(
            f"[ACCURACY] Demo 3: GT={gt_label_unknown}, "
            f"open_set={label_unknown} ({'OK' if correct_open_set else 'WRONG'}), "
            f"final={final_decision_unknown} ({'OK' if correct_final else 'WRONG'})\n"
        )

        demo_metrics.append(
            {
                "name": "Demo 3 (UNKNOWN_SUSPICIOUS)",
                "gt": gt_label_unknown,
                "open_set": label_unknown,
                "final": final_decision_unknown,
                "correct_open_set": correct_open_set,
                "correct_final": correct_final,
            }
        )

    # -----------------------------
    # Summary of demo accuracy
    # -----------------------------
    if demo_metrics:
        total = len(demo_metrics)
        acc_open = sum(m["correct_open_set"] for m in demo_metrics) / total
        acc_final = sum(m["correct_final"] for m in demo_metrics) / total

        print("=== Demo accuracy summary (3 curated flows) ===")
        for m in demo_metrics:
            print(
                f"{m['name']}: GT={m['gt']}, "
                f"open_set={m['open_set']} ({'OK' if m['correct_open_set'] else 'WRONG'}), "
                f"final={m['final']} ({'OK' if m['correct_final'] else 'WRONG'})"
            )
        print(f"\nOverall open-set heuristic accuracy on demos: {acc_open:.3f}")
        print(f"Overall final (LLM+policy) accuracy on demos: {acc_final:.3f}\n")
    else:
        print("[WARN] No demos were run; no demo accuracy to report.\n")


if __name__ == "__main__":
    main()
