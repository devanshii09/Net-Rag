import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Paths / imports ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore


# --- Open-set config ---------------------------------------------------------

UNSEEN_CATS = ["Worms", "Shellcode", "Backdoor"]


# --- Helpers -----------------------------------------------------------------


def make_feature_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Split df into (X, y, numeric_cols, categorical_cols).

    We:
      - Drop obvious non-feature columns: id, label, attack_cat, split
      - Infer categorical vs numeric by dtype
    """
    drop_cols = ["id", "label", "attack_cat", "split"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int).values

    # Simple heuristic: object / category -> categorical, others numeric
    categorical_cols = [
        c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return X, y, numeric_cols, categorical_cols


def build_baseline_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    """
    Build a simple baseline pipeline:
      - OneHotEncode categoricals
      - Passthrough numeric
      - RandomForest classifier
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight=None,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def evaluate_split(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    label: str,
) -> Tuple[float, float]:
    """
    Evaluate trained pipeline on a given split and print metrics.
    """
    y_pred = pipe.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n=== {label} ===")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    return acc, f1


def main() -> None:
    # ---------------------------------------------------------------------
    # 1. Load data and define splits
    # ---------------------------------------------------------------------
    df = load_unsw_nb15_full()

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    # open-set: we train only on "seen" categories (exclude UNSEEN_CATS)
    df_train_seen = df_train[~df_train["attack_cat"].isin(UNSEEN_CATS)].copy()
    df_test_seen = df_test[~df_test["attack_cat"].isin(UNSEEN_CATS)].copy()
    df_test_unseen = df_test[df_test["attack_cat"].isin(UNSEEN_CATS)].copy()

    print("[INFO] Baseline open-set config:")
    print(f"  UNSEEN_CATS:        {UNSEEN_CATS}")
    print(f"  Train (all):        {len(df_train)}")
    print(f"  Train (seen only):  {len(df_train_seen)}")
    print(f"  Test (all):         {len(df_test)}")
    print(f"  Test (seen only):   {len(df_test_seen)}")
    print(f"  Test (unseen only): {len(df_test_unseen)}")

    # ---------------------------------------------------------------------
    # 2. Build features / targets
    # ---------------------------------------------------------------------
    X_train, y_train, num_cols, cat_cols = make_feature_target(df_train_seen)
    X_test_all, y_test_all, _, _ = make_feature_target(df_test)
    X_test_seen, y_test_seen, _, _ = make_feature_target(df_test_seen)
    X_test_unseen, y_test_unseen, _, _ = make_feature_target(df_test_unseen)

    print("\n[INFO] Feature columns:")
    print(f"  Numeric:      {len(num_cols)} columns")
    print(f"  Categorical:  {len(cat_cols)} columns")

    # ---------------------------------------------------------------------
    # 3. Build and train baseline model
    # ---------------------------------------------------------------------
    print("\n[INFO] Training RandomForest baseline on TRAIN SEEN ONLY...")
    pipe = build_baseline_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # 4. Evaluate on all / seen / unseen test splits
    # ---------------------------------------------------------------------
    results = {}

    acc_all, f1_all = evaluate_split(pipe, X_test_all, y_test_all, label="ALL test")
    results["ALL"] = (acc_all, f1_all)

    acc_seen, f1_seen = evaluate_split(pipe, X_test_seen, y_test_seen, label="SEEN only")
    results["SEEN"] = (acc_seen, f1_seen)

    acc_unseen, f1_unseen = evaluate_split(pipe, X_test_unseen, y_test_unseen, label="UNSEEN only")
    results["UNSEEN"] = (acc_unseen, f1_unseen)

    # ---------------------------------------------------------------------
    # 5. Small summary
    # ---------------------------------------------------------------------
    print("\n[SUMMARY] Baseline RandomForest (train on seen only):")
    print("Split    Acc       F1")
    for split, (acc, f1) in results.items():
        print(f"{split:6s}  {acc:0.6f}  {f1:0.6f}")

def train_open_set_baseline(unseen_cats=UNSEEN_CATS):
    df = load_unsw_nb15_full()

    df_train = df[df["split"] == "train"].copy()
    df_test_all = df[df["split"] == "test"].copy()

    df_train_seen = df_train[~df_train["attack_cat"].isin(unseen_cats)].copy()
    df_test_seen = df_test_all[~df_test_all["attack_cat"].isin(unseen_cats)].copy()
    df_test_unseen = df_test_all[df_test_all["attack_cat"].isin(unseen_cats)].copy()

    # reuse the SAME logic you used in main()
    X_train_seen, y_train_seen, numeric_cols, cat_cols = make_feature_target(df_train_seen)

    pipe = build_baseline_pipeline(numeric_cols, cat_cols)
    pipe.fit(X_train_seen, y_train_seen)

    return {
        "df": df,
        "df_train_seen": df_train_seen,
        "df_test": df_test_all,
        "df_test_seen": df_test_seen,
        "df_test_unseen": df_test_unseen,
        "pipeline": pipe,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
    }

if __name__ == "__main__":
    main()