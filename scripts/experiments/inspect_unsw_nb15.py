import sys
from pathlib import Path

# Make sure src/ is on sys.path if you didn't do editable install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore


def main():
    df = load_unsw_nb15_full()

    print("[INFO] Loaded UNSW-NB15 full dataset")
    print(f"[INFO] Shape: {df.shape}")
    print("[INFO] Columns:", df.columns.tolist())

    print("\n[INFO] Label distribution:")
    print(df["label"].value_counts(dropna=False))

    if "attack_cat" in df.columns:
        print("\n[INFO] Attack category distribution (top 10):")
        print(df["attack_cat"].value_counts().head(10))

    print("\n[INFO] Sample rows:")
    print(df.head(5))


if __name__ == "__main__":
    main()