import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from net_rag.data.unsw_nb15_loader import load_unsw_nb15_full  # type: ignore
from net_rag.models.open_set import (  # type: ignore
    train_open_set_baseline,
    evaluate_open_set_baseline,
)


def main():
    df = load_unsw_nb15_full()

    # Pick one or more categories to treat as "unseen"
    # Start small: rare but meaningful category, e.g. 'Worms' or 'Shellcode'
    unseen_attack_cats = ["Worms"]

    print("[INFO] Training open-set baseline (hiding categories:", unseen_attack_cats, ")")
    model = train_open_set_baseline(df, unseen_attack_cats)

    print("[INFO] Evaluating open-set baseline...")
    _ = evaluate_open_set_baseline(df, model, unseen_attack_cats)


if __name__ == "__main__":
    main()