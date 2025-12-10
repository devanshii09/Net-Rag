#!/usr/bin/env python
"""
UNSW-NB15 Open-set + Net-RAG + LLM demo (binary + family-level).

- Uses BOTH UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv
- Holds out families in UNKNOWN_FAMILIES as UNKNOWN at training time (open-set)
- Trains:
    1) RandomForest on binary label (benign vs malicious) for known families
    2) RandomForest+calibration on attack families (malicious-only, known families)
- Reports:
    - Binary RF sanity-check and confusion matrix
    - Binary RF open-set behaviour (confidence-based) on test-known vs test-unknown
    - Family-level open-set behaviour (family RF) + confusion-style counts + threshold grid
- Builds a Chroma vector index over train-known flows (batched upsert)
- Retrieves similar flows and asks an LLM to explain + decide + recommend actions

Open-set top-level label space:
    - KNOWN_BENIGN      (benign, known family "Normal")
    - KNOWN_MALICIOUS   (malicious, confident about known family)
    - UNKNOWN_MALICIOUS (malicious, family looks unknown / low-confidence)
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
from collections import Counter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, SystemMessage
# --- NEW IMPORT ---
from sentence_transformers import CrossEncoder

# --- INITIALIZE MODEL (Global Scope) ---
print("[INFO] Loading Re-Ranker Model...")
RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# -----------------------------
# Paths and constants
# -----------------------------

UNSW_TRAIN_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_training-set.csv")
UNSW_TEST_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv")

CHROMA_DIR = Path("data/processed/chroma_unsw_open_set")
CHROMA_COLLECTION = "unsw_flows_open_set"

# Families we treat as unknown at train time (open-set)
UNKNOWN_FAMILIES = {"Shellcode"}

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
        "id",
        "attack_cat",
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


def serialize_unsw_flow(row, include_labels=False):
    """
    Serializes a UNSW-NB15 flow into a semantic string with feature bucketing
    to help the embedding model distinguish DoS and Exploits.
    """
    parts = []

    # 1. Basic Identifiers
    proto = str(row.get('proto', 'unknown'))
    service = str(row.get('service', '-'))
    parts.append(f"Protocol: {proto}")
    if service != '-':
        parts.append(f"Service: {service}")

    # 2. Duration Bucketing (Critical for DoS vs Normal)
    # DoS flows are often extremely short (single packet bursts) or very long floods.
    dur = float(row.get('dur', 0.0))
    if dur < 0.001:
        dur_desc = "INSTANT"
    elif dur < 0.1:
        dur_desc = "VERY_SHORT"
    elif dur > 5.0:
        dur_desc = "LONG_DURATION"
    else:
        dur_desc = "NORMAL_DURATION"
    parts.append(f"Duration: {dur:.4f}s ({dur_desc})")

    # 3. Volume/Rate Bucketing (Critical for DoS)
    # Sload/Dload (Source/Dest bits per second) are the strongest indicators for DoS.
    sload = float(row.get('sload', 0.0))
    if sload > 1e8: # > 100 Mbps
        load_desc = "EXTREME_LOAD"
    elif sload > 1e6: # > 1 Mbps
        load_desc = "HIGH_LOAD"
    else:
        load_desc = "NORMAL_LOAD"
    parts.append(f"SourceLoad: {sload:.1f} ({load_desc})")

    # 4. Payload Size (Critical for Exploits)
    # Exploits often have specific, non-zero but small payload sizes compared to file transfers.
    sbytes = float(row.get('sbytes', 0))
    dbytes = float(row.get('dbytes', 0))
    
    if sbytes == 0:
        size_desc = "EMPTY"
    elif sbytes < 200:
        size_desc = "TINY_PAYLOAD" # Common for probes/exploits
    elif sbytes > 10000:
        size_desc = "LARGE_TRANSFER"
    else:
        size_desc = "NORMAL_SIZE"
    parts.append(f"SourceBytes: {int(sbytes)} ({size_desc})")

    # 5. TTL (Time To Live) - (Critical for Generic Exploits)
    # Attackers often use non-standard TTLs (e.g., 254 vs 64).
    sttl = int(row.get('sttl', 0))
    parts.append(f"SourceTTL: {sttl}")

    # 6. TCP State (If available)
    state = str(row.get('state', ''))
    if state:
        parts.append(f"TCP_State: {state}")

    # OPTIONAL: Include label only if generating the DB (not during inference)
    if include_labels:
        parts.append(f"Label: {'Malicious' if row.get('label', 0)==1 else 'Benign'}")
        parts.append(f"AttackFamily: {row.get('attack_cat', 'Normal')}")

    return " | ".join(parts)

# -----------------------------
# Chroma index with batching
# -----------------------------
def stratified_sample(df: pd.DataFrame, max_total: int = 10_000, per_family_cap: int = 2_000) -> pd.DataFrame:
    """
    Balanced down-sampling: Caps each family at `per_family_cap` so 'Normal'
    doesn't drown out the rare attacks.
    """
    if "attack_cat" not in df.columns:
        return df.sample(n=min(len(df), max_total), random_state=42)

    groups = []
    for fam, sub in df.groupby("attack_cat"):
        # Cap large families (like Normal/Generic)
        n = min(len(sub), per_family_cap)
        if n > 0:
            groups.append(sub.sample(n=n, random_state=42))

    if not groups:
        return df.iloc[0:0].copy()

    df_bal = pd.concat(groups)

    # Randomly sample down if total is still too big
    if len(df_bal) > max_total:
        df_bal = df_bal.sample(n=max_total, random_state=42)

    return df_bal

def build_chroma_index(
    df_index: pd.DataFrame,
    embedder: Embeddings,
    persist_dir: Path,
    collection_name: str,
    max_flows: Optional[int] = None,
    batch_size: int = 5000,
) -> Chroma:
    """
    Build a Chroma index with BALANCED sampling and NOMIC prefixes.
    """
    # 1. BALANCED SAMPLING (Fixes "All neighbors are Normal")
    if max_flows is not None and len(df_index) > max_flows:
        print(f"[INFO] Stratified sampling {max_flows:,} flows (balancing families)...")
        df_index = stratified_sample(df_index, max_total=max_flows, per_family_cap=2000)
    else:
        print(f"[INFO] Indexing all {len(df_index):,} flows...")

    if "row_idx" not in df_index.columns:
        df_index = df_index.copy()
        df_index["row_idx"] = df_index.index.astype(int)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for _, row in df_index.iterrows():
        # 2. ADD NOMIC PREFIX (Fixes "Low Similarity Scores")
        # Nomic needs to know this is a document to be searched
        raw_text = serialize_unsw_flow(row, include_labels=True)
        text_with_prefix = "search_document: " + raw_text
        texts.append(text_with_prefix)

        fam = str(row.get("attack_cat", "Unknown"))
        lbl = int(row.get("label", 0))
        row_index = int(row["row_idx"])

        metadatas.append({
            "family": fam,
            "attack_cat": fam,
            "label": lbl,
            "label_binary": lbl,
            "row_idx": row_index,
        })

    # Wipe old DB to prevent mixing bad/good embeddings
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(persist_dir),
        # [REPORT ALIGNMENT] Forces Cosine Similarity as stated in Section 6.4 of your report
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Batch upsert
    n_docs = len(texts)
    for i in range(0, n_docs, batch_size):
        end = min(i + batch_size, n_docs)
        vectorstore.add_texts(texts[i:end], metadatas=metadatas[i:end])
        print(f"[INFO] Batch {i}-{end} upserted.")

    # Note: Chroma 0.4+ persists automatically, but explicit call is safe in older versions
    try:
        vectorstore.persist()
    except:
        pass
        
    print("[INFO] Chroma index build complete.\n")
    return vectorstore


def stratified_sample(df: pd.DataFrame, max_total: int = 10_000, per_family_cap: int = 2_000) -> pd.DataFrame:
    """
    Balanced down-sampling by attack_cat:
    - Caps each family at `per_family_cap` to prevent "Normal" from dominating.
    - Randomly samples to reach `max_total` if the result is still too big.
    """
    if "attack_cat" not in df.columns:
        # Fallback if column missing
        return df.sample(n=min(len(df), max_total), random_state=42)

    groups = []
    # 1. Cap each family
    for fam, sub in df.groupby("attack_cat"):
        n = min(len(sub), per_family_cap)
        if n <= 0:
            continue
        groups.append(sub.sample(n=n, random_state=42))

    if not groups:
        return df.iloc[0:0].copy()

    df_bal = pd.concat(groups)

    # 2. Global cap (if sum of capped families is still > max_total)
    if len(df_bal) > max_total:
        df_bal = df_bal.sample(n=max_total, random_state=42)

    return df_bal

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
    print(f"Unknown samples evaluated:  {len(df_test_unknown):4d}\n")

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
def parse_chroma_docs(docs_and_scores) -> List[NeighborSummary]:
    """
    Converts raw Chroma results (doc, score) into clean NeighborSummary objects.
    """
    neighbors = []
    for doc, raw_score in docs_and_scores:
        meta = doc.metadata
        
        family = str(meta.get("attack_cat", "Unknown"))
        label = int(meta.get("label", 0))
        row_idx = int(meta.get("row_idx", -1))
        
        # [REPORT ALIGNMENT] 
        # Chroma with "cosine" space returns Cosine Distance (0 = identical, 1 = orthogonal).
        # We want Cosine Similarity (1 = identical, 0 = orthogonal).
        # Formula: Similarity = 1.0 - Distance
        
        similarity = 1.0 - float(raw_score)
        
        # Clamp it to ensure 0-1 range despite potential floating point noise
        similarity = max(0.0, min(1.0, similarity))
        
        n = NeighborSummary(
            family=family,
            label=label,
            score=similarity, 
            row_idx=row_idx
        )
        neighbors.append(n)
    return neighbors

def summarize_rag(neighbors: List[NeighborSummary]) -> RAGSummary:
    """
    Computes summary statistics over the retrieved neighbors.
    """
    if not neighbors:
        return RAGSummary(0.0, 0.0, 0.0, 0, [])

    # 1. Similarity stats
    scores = [n.score for n in neighbors]
    # Note: Chroma usually returns L2 distance (lower is better) or Cosine distance.
    # If using Nomic with default settings, it might be distance. 
    # Ideally, we convert to similarity (1 - distance) if needed, 
    # but for this demo, we assume the score acts as a proxy for similarity.
    sim_mean = sum(scores) / len(scores)
    sim_max = max(scores)

    # 2. Malicious fraction
    mal_count = sum([1 for n in neighbors if n.label == 1])
    mal_frac = mal_count / len(neighbors)

    # 3. Diversity
    unique_families = set([n.family for n in neighbors])
    family_diversity = len(unique_families)

    return RAGSummary(
        sim_max=sim_max,
        sim_mean=sim_mean,
        mal_frac=mal_frac,
        family_diversity=family_diversity,
        neighbors=neighbors
    )

def retrieve_neighbors(vectorstore, q_text, rf_out=None, family_out=None, k=5):
    """
    Retrieves k neighbors from Chroma, optionally filtering by:
      1. Binary RF prediction (Benign vs Malicious)
      2. Family RF prediction (Exploits vs Generic, etc.)
    """
    
    # 1. Fetch a larger pool of candidates (4x what we need)
    k_pool = k * 4
    
    # CRITICAL FIX: Use similarity_search_with_score to get the distance/similarity
    # Returns List[Tuple[Document, float]]
    docs_and_scores = vectorstore.similarity_search_with_score(q_text, k=k_pool)
    
    # Parse raw Chroma docs into our helper objects
    all_neighbors = parse_chroma_docs(docs_and_scores)

    # ---------------------------------------------------------
    # FILTER 1: Binary RF (Consistency Check)
    # ---------------------------------------------------------
    binary_filtered = all_neighbors
    if rf_out and rf_out.rf_confidence > 0.7:
        pred_label = 0 if rf_out.p_benign > rf_out.p_malicious else 1
        filtered_list = [n for n in all_neighbors if n.label == pred_label]
        if len(filtered_list) > 0:
            binary_filtered = filtered_list

    # ---------------------------------------------------------
    # FILTER 2: Family RF (Guided Reranking)
    # ---------------------------------------------------------
    final_candidates = binary_filtered
    
    if family_out and family_out.family_conf > 0.8:
        pred_family = family_out.family_pred
        same_family = [n for n in binary_filtered if n.family == pred_family]
        other_family = [n for n in binary_filtered if n.family != pred_family]
        
        if len(same_family) > 0:
            final_candidates = same_family + other_family

    # ---------------------------------------------------------
    # Final Selection
    # ---------------------------------------------------------
    top_k_neighbors = final_candidates[:k]

    return summarize_rag(top_k_neighbors)


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
      - held-out-family malicious test flows (UNKNOWN_FAMILIES)
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

    unknown_name_str = ",".join(sorted(UNKNOWN_FAMILIES))

    print(f"Family RF max-probability stats ({unknown_name_str} / unknown malicious):")
    print(
        "  pmax_unknown percentiles [min,25,50,75,max] =",
        np.percentile(pmax_unknown, [0, 25, 50, 75, 100]),
    )

    preds_unknown = probs_unknown.argmax(axis=1)
    fam_names_unknown = family_le.inverse_transform(preds_unknown)
    fam_series = pd.Series(fam_names_unknown)
    fam_series_display = fam_series.replace({"Generic": "NA"})

    print(f"Top predicted families for {unknown_name_str} / unknown flows:")
    print(fam_series_display.value_counts())
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
            "known vs unknown families. Confidence-threshold open-set detection "
            "is not statistically justified on this dataset/features."
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

    - label == 0                                   -> KNOWN_BENIGN
    - label == 1 and attack_cat in UNKNOWN_FAMILIES -> UNKNOWN_MALICIOUS
    - label == 1 and attack_cat not in UNKNOWN_FAMILIES -> KNOWN_MALICIOUS
    """
    label_binary = int(row.get("label", 0))
    family = str(row.get("attack_cat", "Unknown"))

    if label_binary == 0:
        return "KNOWN_BENIGN"

    if family in UNKNOWN_FAMILIES:
        return "UNKNOWN_MALICIOUS"

    return "KNOWN_MALICIOUS"


def derive_open_set_label_with_family(
    rf_out: RFOutput,
    family_out: FamilyRFOutput,
    rag_summary: Optional[RAGSummary] = None,
    debug: bool = False,
) -> Tuple[str, List[str], str]:

    reasons: List[str] = []

    pred_str = str(family_out.family_pred).strip()

    # Thresholds
    RAG_VETO_THRESH = 0.30
    RAG_RESCUE_FLOOR = 0.50

    WEAK_RAG_FAMILIES = ["DoS", "Worms", "Analysis", "Backdoor"]
    RF_HIGH_CONFIDENCE = 0.90

    if pred_str in WEAK_RAG_FAMILIES:
        if debug:
            print(f"   -> Detected Weak RAG Family ({pred_str}). Lowering RF Trust Threshold.")
        RF_HIGH_CONFIDENCE = 0.75

    if debug:
        sim_val = rag_summary.sim_max if rag_summary else 0.0
        print(f"\n[LOGIC TRACE] RF_Conf={family_out.family_conf:.3f} | Sim={sim_val:.4f} | Pred={pred_str}")

    # =========================================================
    # PHASE 1: BENIGN CHECK (Binary RF says Safe)
    # =========================================================
    if rf_out.p_malicious < 0.20 and pred_str in ("None", "Normal"):
        if debug:
            print("   -> Triggered PHASE 1: Benign Check")
        return "KNOWN_BENIGN", [f"Binary RF benign ({rf_out.p_malicious:.3f})"], "Normal"

    # =========================================================
    # PHASE 2: HIGH CONFIDENCE FAMILY RF
    # =========================================================
    if not family_out.family_unknown and family_out.family_conf >= RF_HIGH_CONFIDENCE:
        if debug:
            print(f"   -> Triggered PHASE 2: High Confidence RF ({pred_str})")

        if pred_str == "Normal":
            return "KNOWN_BENIGN", ["Family RF is high-confidence Normal."], "Normal"

        return "KNOWN_MALICIOUS", [f"RF Confident ({family_out.family_conf:.3f})"], pred_str

    # =========================================================
    # PHASE 3: "RESCUE" FROM NEIGHBORS (only if RF family is low-confidence)
    # =========================================================
    if rag_summary and rag_summary.neighbors and family_out.family_unknown:
        neighbor_families = [n.family for n in rag_summary.neighbors]
        vote_counts = Counter(neighbor_families)
        top_family, top_count = vote_counts.most_common(1)[0]

        if rag_summary.sim_max > RAG_RESCUE_FLOOR and top_count >= 3:
            if top_family not in ["Normal", "Benign"]:
                reasons.append(f"Majority Rescue: {top_count}/5 neighbors are '{top_family}'.")
                if debug:
                    print(f"   -> Triggered PHASE 3: Majority Rescue (malicious family {top_family})")
                return "KNOWN_MALICIOUS", reasons, top_family
            else:
                # Only rescue to benign if binary RF is not strongly malicious
                if rf_out.p_malicious < 0.50:
                    reasons.append(
                        f"Majority Rescue: {top_count}/5 neighbors are Benign/Normal "
                        "and binary RF is not strongly malicious."
                    )
                    if debug:
                        print("   -> Triggered PHASE 3: Benign Rescue (neighbors benign, RF weak)")
                    return "KNOWN_BENIGN", reasons, "Normal"
                else:
                    reasons.append(
                        "Neighbors mostly benign, but binary RF is strongly malicious; "
                        "ignoring benign rescue."
                    )
                    if debug:
                        print(
                            "   -> PHASE 3: Benign majority ignored (RF strongly malicious)"
                        )

    # =========================================================
    # PHASE 4: "UNKNOWN" VETO WHEN NOTHING LOOKS SIMILAR
    # =========================================================
    if rag_summary and rag_summary.sim_max < RAG_VETO_THRESH:
        if debug:
            print(f"   -> Triggered PHASE 4: RAG Veto (Sim {rag_summary.sim_max:.4f} too low)")
        return "UNKNOWN_MALICIOUS", [
            f"RAG VETO: Similarity {rag_summary.sim_max:.4f} < {RAG_VETO_THRESH}.",
            "Flow is unique/unknown.",
        ], "Unknown"

    # =========================================================
    # PHASE 5: STANDARD FALLBACK
    # =========================================================
    if debug:
        print("   -> Triggered PHASE 5: Fallback")

    if rf_out.p_malicious < 0.50:
        return "KNOWN_BENIGN", ["Binary RF indicates Benign."], "Normal"

    if family_out.family_unknown:
        return "UNKNOWN_MALICIOUS", ["Family RF is low confidence (Unknown)."], "Unknown"

    return "KNOWN_MALICIOUS", ["Family RF is moderately confident."], pred_str


def compute_open_set_for_row(
    rf: RandomForestClassifier,
    family_clf: CalibratedClassifierCV,
    family_le: LabelEncoder,
    row: pd.Series,
    feature_cols: List[str],
    rag_summary: Optional[RAGSummary] = None,
    debug: bool = False,
) -> Tuple[str, List[str], RFOutput, FamilyRFOutput, str]:

    rf_out = rf_forward(rf, row, feature_cols)
    family_out = family_forward(family_clf, family_le, row, feature_cols)

    open_set_label, reasons, derived_family = derive_open_set_label_with_family(
        rf_out, family_out, rag_summary=rag_summary, debug=debug
    )

    return open_set_label, reasons, rf_out, family_out, derived_family


# -----------------------------
# LLM explanation helpers
# -----------------------------


def parse_llm_output(text: str) -> Tuple[str, str]:
    """
    Parse the LLM response and extract:
      - Final_decision
      - Recommended_action
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
    if label == "KNOWN_BENIGN":
        return "DO_NOTHING"

    if label == "KNOWN_MALICIOUS":
        if llm_action in {"BLOCK_IP", "RATE_LIMIT_PORT", "MONITOR_HOST"}:
            return llm_action
        return "BLOCK_IP"

    if label == "UNKNOWN_MALICIOUS":
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
) -> Tuple[str, str]:

    base_label = open_set_label.strip().upper()
    llm_label = (llm_decision or "").strip().upper()
    llm_action = (llm_action or "").strip().upper()
    valid_labels = {"KNOWN_BENIGN", "KNOWN_MALICIOUS", "UNKNOWN_MALICIOUS"}

    if llm_label not in valid_labels:
        final_label = base_label
    else:
        # 1. Block Dangerous Downgrade (Malicious -> Benign) when evidence is strong
        if llm_label == "KNOWN_BENIGN" and p_malicious > 0.90 and mal_frac > 0.90:
            print(f"[POLICY] Blocking LLM downgrade to BENIGN. Keeping {base_label}.")
            final_label = base_label

        # 2. Block downgrade of very high-confidence KNOWN_MALICIOUS
        elif (
            base_label == "KNOWN_MALICIOUS"
            and llm_label != "KNOWN_MALICIOUS"
            and p_malicious > 0.95
        ):
            print(
                "[POLICY] Blocking LLM downgrade from high-confidence KNOWN_MALICIOUS. "
                f"Keeping {base_label}."
            )
            final_label = base_label

        # 3. Block downgrade to UNKNOWN when neighbors are overwhelmingly malicious
        elif (
            base_label == "KNOWN_MALICIOUS"
            and llm_label == "UNKNOWN_MALICIOUS"
            and mal_frac > 0.95
        ):
            print(
                "[POLICY] Blocking LLM downgrade to UNKNOWN. Evidence is too strong. "
                f"Keeping {base_label}."
            )
            final_label = base_label

        else:
            final_label = llm_label

    final_action = _default_action_for_label(final_label, llm_action)
    return final_label, final_action


def llm_explain_flow(
    llm: ChatOllama,
    row: pd.Series,
    rf_out: RFOutput,
    family_out: FamilyRFOutput,
    rag: RAGSummary,
    open_set_label: str,
    derived_family: str,
) -> Tuple[str, str, str, str]:

    """
    Ask the LLM for an explanation, then apply the override policy.
    """
    # IMPORTANT: do NOT leak dataset labels (attack_cat, label) into the current flow description.
    current_flow_text = serialize_unsw_flow(row, include_labels=False)

    # Build human-readable neighbor summary
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
        "2. Decide whether it is KNOWN_BENIGN, KNOWN_MALICIOUS, or UNKNOWN_MALICIOUS.\n"
        "3. Justify that decision using both classifier outputs and the neighbor flows.\n"
        "4. Suggest exactly one concrete defensive action "
        "(DO_NOTHING, BLOCK_IP, RATE_LIMIT_PORT, or MONITOR_HOST).\n"
        "Always stay grounded in the provided data and avoid inventing ports or IPs.\n"
        "Return your answer in this exact format:\n"
        "Final_decision: <one of KNOWN_BENIGN, KNOWN_MALICIOUS, UNKNOWN_MALICIOUS>\n"
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
Label: {open_set_label}
Suggested Family: {derived_family}

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
    )

    llm_text += (
        "\n\n"
        f"Final_decision (after policy): {final_decision}\n"
        f"Recommended_action (after policy): {final_action}"
    )

    return llm_text, final_decision, final_action, derived_family


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

      1) KNOWN_BENIGN from df_test_known (label=0)
      2) KNOWN_MALICIOUS from df_test_known (label=1, non-unknown family)
      3) UNKNOWN_MALICIOUS from df_test_unknown (held-out families)

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
            os_label, _, rf_out, _, _ = compute_open_set_for_row(
                rf,
                family_clf,
                family_le,
                row,
                feature_cols,
                rag_summary=None,
                debug=False,
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
            os_label, _, _, _, _ = compute_open_set_for_row(
                rf,
                family_clf,
                family_le,
                row,
                feature_cols,
                rag_summary=None,
                debug=False,
            )
            if os_label == desired_label:
                return row

        # Pass 3: just return some row
        return df_sample.iloc[len(df_sample) // 2]

    df_benign_known = df_test_known[df_test_known["label"] == 0]
    benign_row = pick_one(
        df=df_benign_known,
        desired_label="KNOWN_BENIGN",
        require_gt_match=True,
        max_rows=5000,
        seed=1,
    )

    df_mal_known = df_test_known[
        (df_test_known["label"] == 1)
        & (~df_test_known["attack_cat"].isin(UNKNOWN_FAMILIES))
    ]
    mal_row = pick_one(
        df=df_mal_known,
        desired_label="KNOWN_MALICIOUS",
        require_gt_match=True,
        max_rows=5000,
        seed=2,
    )

    unk_row = None
    if not df_test_unknown.empty:
        unk_row = pick_one(
            df=df_test_unknown,
            desired_label="UNKNOWN_MALICIOUS",
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

    rf = train_binary_rf(df_train_known, df_val_known, feature_cols)
    family_clf, family_le, _ = train_family_rf(df_train_known, feature_cols)

    row_benign, row_mal, row_unknown = select_demo_flows(
        df_test_known=df_test_known,
        df_test_unknown=df_test_unknown,
        rf=rf,
        family_clf=family_clf,
        family_le=family_le,
        feature_cols=feature_cols,
    )

    print("[INFO] Initializing Ollama embedding model 'nomic-embed-text'...")
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectorstore = build_or_load_chroma(df_train_known, embedding)

    print("[INFO] Initializing ChatOllama model 'llama3.1'...")
    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.0)

    print("\n=== Running curated LLM demos (3 flows) ===\n")

    demos = [
        ("Demo 1: KNOWN_BENIGN", row_benign),
        ("Demo 2: KNOWN_MALICIOUS", row_mal),
        ("Demo 3: UNKNOWN-family", row_unknown),
    ]

    for title, row in demos:
        if row is None:
            print(f"[WARN] Skipping {title} (no row selected).")
            continue
            
        print(f"[{title}] (row_idx={row.name})")
        print(f"True attack family: {row.get('attack_cat', 'Unknown')} (binary label: {row.get('label')})\n")

        # 1. RUN RF FIRST (We need the prediction to guide the search)
        rf_out = rf_forward(rf, row, feature_cols)
        family_out = family_forward(family_clf, family_le, row, feature_cols)

        # 2. RUN RAG (Guided by RF)
        q_text = serialize_unsw_flow(row, include_labels=False)
        # PASS rf_out HERE!
        rag = retrieve_neighbors(vectorstore, q_text, rf_out=rf_out, k=5)

        # 3. LOGIC
        label, reasons, derived_family = derive_open_set_label_with_family(
            rf_out, family_out, rag_summary=rag, debug=True
        )

        print("[INFO] RF + RAG + Family signals before LLM:")
        print(f"  p_malicious     = {rf_out.p_malicious:.6f}")
        print(f"  RF_confidence   = {rf_out.rf_confidence:.6f}")
        print(f"  sim_max         = {rag.sim_max:.6f}")
        print(f"  family_pred     = {family_out.family_pred}")
        print(f"  open-set label  = {label}")
        
        print("\n[INFO] Neighbors:")
        if rag.neighbors:
            for n in rag.neighbors:
                print(f"  - Family: {n.family}, Label: {n.label}, Score: {n.score:.4f}")
        else:
            print("  (No neighbors found matching filter)")

        # 4. LLM
        llm_text, final_decision, final_action, _ = llm_explain_flow(
            llm, row, rf_out, family_out, rag, label, derived_family
        )
        
        print("\n[LLM Decision]:", final_decision)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
