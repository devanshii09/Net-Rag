import pandas as pd

cases = pd.read_csv("data/processed/unsw_rag_vs_baseline_cases.csv")

print(cases.head())

print("\nCase type counts:")
print(cases["case_type"].value_counts())

print("\nCase types by attack_cat:")
print(cases.groupby(["attack_cat", "case_type"]).size())

# ---------------- NEW PART: sample interesting flows ----------------

# 1) Where RAG is right and RF is wrong, on NORMAL traffic (likely RF over-flagging benign)
rag_norm = cases[
    (cases["case_type"] == "rag_only_correct")
    & (cases["attack_cat"] == "Normal")
]

print("\n[Sample] rag_only_correct & Normal (RAG fixes RF false positives):")
print(rag_norm.sample(5, random_state=0)[["id", "label", "rf_pred", "rag_pred"]])

# 2) Where RF is right and RAG is wrong, on FUZZERS (RAG misses attacks RF catches)
rf_fuzzers = cases[
    (cases["case_type"] == "rf_only_correct")
    & (cases["attack_cat"] == "Fuzzers")
]

print("\n[Sample] rf_only_correct & Fuzzers (RF catches attacks RAG misses):")
print(rf_fuzzers.sample(5, random_state=0)[["id", "label", "rf_pred", "rag_pred"]])