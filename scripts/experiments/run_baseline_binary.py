import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.models.baseline import train_baseline_binary, evaluate_baseline_binary  # type: ignore


def main():
    df = load_unsw_nb15_full()

    print("[INFO] Training baseline binary classifier on UNSW-NB15...")
    model = train_baseline_binary(df)

    print("[INFO] Evaluating baseline binary classifier...")
    evaluate_baseline_binary(df, model)


if __name__ == "__main__":
    main()