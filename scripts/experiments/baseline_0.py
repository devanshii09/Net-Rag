#!/usr/bin/env python
"""
UNSW-NB15 CLOSED-SET baseline (no open-set, no RAG, no LLM).

- Uses UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv
- Trains:
    1) RandomForest on binary label (benign vs malicious)
    2) RandomForest on attack_cat (multi-class family) using only malicious rows
- Evaluates:
    - Binary RF on the test CSV (closed-set)
    - Family RF on malicious rows in the test CSV (closed-set)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Paths
# -----------------------------

UNSW_TRAIN_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_training-set.csv")
UNSW_TEST_CSV = Path("data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv")


# -----------------------------
# Data loading / preprocessing
# -----------------------------

def load_unsw_train_test(
    train_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Loading TRAIN from {train_path}")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"[INFO] Loading TEST  from {test_path}")
    df_test = pd.read_csv(test_path, low_memory=False)

    print(f"[INFO] Train rows: {len(df_train):,}")
    print(f"[INFO] Test  rows: {len(df_test):,}")

    # Basic sanity checks
    for df_name, df in [("train", df_train), ("test", df_test)]:
        if "attack_cat" not in df.columns:
            raise ValueError(f"{df_name} CSV missing 'attack_cat' column.")
        if "label" not in df.columns:
            raise ValueError(f"{df_name} CSV missing 'label' column.")

        # Normalize types
        df["attack_cat"] = df["attack_cat"].fillna("Unknown").astype(str)
        df["label"] = df["label"].astype(int)

    return df_train, df_test


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """Pick numeric feature cols, dropping label-like things."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = {
        "label",
        "label_binary",
        "row_idx",
    }

    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    print(f"[INFO] Using {len(feature_cols)} numeric feature columns.")
    return feature_cols


# -----------------------------
# Binary RF: benign vs malicious
# -----------------------------

def train_binary_rf(
    df_train: pd.DataFrame,
    feature_cols: List[str],
) -> RandomForestClassifier:
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    print("[INFO] Training RandomForest (binary benign vs malicious)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    return rf


def eval_binary_rf(
    rf: RandomForestClassifier,
    df_test: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    y_pred = rf.predict(X_test)

    print("\n=== Binary RF: closed-set performance on TEST ===")
    print(classification_report(y_test, y_pred, digits=3, target_names=["benign", "malicious"]))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")


# -----------------------------
# Family RF: attack_cat (malicious only)
# -----------------------------

def train_family_rf(
    df_train: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[RandomForestClassifier, LabelEncoder, List[str]]:
    """
    Train a multi-class RF on attack_cat using only malicious rows.
    This is CLOSED-SET: no families are held out here.
    """
    df_mal_train = df_train[df_train["label"] == 1].copy()
    if df_mal_train.empty:
        raise ValueError("No malicious rows in training data.")

    X_train = df_mal_train[feature_cols].values
    y_family = df_mal_train["attack_cat"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_family)
    family_classes = list(le.classes_)
    print(f"[INFO] Family classes (closed-set) seen in TRAIN: {family_classes}")

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    print("[INFO] Training RandomForest (family multi-class, malicious only)...")
    clf.fit(X_train, y_enc)
    return clf, le, family_classes


def eval_family_rf(
    clf: RandomForestClassifier,
    le: LabelEncoder,
    df_test: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    """
    Evaluate family RF on malicious TEST rows (same families as train, so closed-set).
    """
    df_mal_test = df_test[df_test["label"] == 1].copy()
    if df_mal_test.empty:
        print("[WARN] No malicious rows in TEST; skipping family RF eval.")
        return

    X_test = df_mal_test[feature_cols].values
    y_true_family = df_mal_test["attack_cat"].values
    y_true_enc = le.transform(y_true_family)  # will fail if unseen family appears

    y_pred_enc = clf.predict(X_test)
    y_pred_family = le.inverse_transform(y_pred_enc)

    print("\n=== Family RF: closed-set performance on malicious TEST rows ===")
    print(
        classification_report(
            y_true_enc,
            y_pred_enc,
            digits=3,
            target_names=list(le.classes_),
        )
    )

    # Optional: confusion matrix (can be big if many families)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    print("Family confusion matrix shape:", cm.shape)
    print("NOTE: you can inspect cm or per-class metrics above.")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    df_train, df_test = load_unsw_train_test(UNSW_TRAIN_CSV, UNSW_TEST_CSV)

    feature_cols = choose_feature_columns(df_train)

    # --- Binary RF baseline ---
    rf_bin = train_binary_rf(df_train, feature_cols)
    eval_binary_rf(rf_bin, df_test, feature_cols)

    # --- Family RF baseline (malicious only) ---
    family_clf, family_le, family_classes = train_family_rf(df_train, feature_cols)
    eval_family_rf(family_clf, family_le, df_test, feature_cols)

    print("\n[INFO] Closed-set baselines completed.")
    print("[INFO] If these numbers look sane, THEN we can reintroduce open-set / RAG / LLM on top.")


if __name__ == "__main__":
    main()