# Net-RAG: Open-Set Intrusion Detection with RAG + LLM on UNSW-NB15

This repository contains the code for **Net‑RAG**, an open‑set intrusion detection demo on the **UNSW‑NB15** dataset. It combines:

- A **binary Random Forest** for benign vs malicious.
- A **family‑level Random Forest + calibration** for known attack families.
- An **open‑set heuristic** that can output: `KNOWN_BENIGN`, `KNOWN_MALICIOUS`, `UNKNOWN_MALICIOUS`.
- A **Chroma** vector index over historical flows (Net‑RAG).
- A local **LLM via Ollama** for per‑flow explanations and recommended defensive actions.
- A **Streamlit UI** to explore individual flows, neighbors, and LLM explanations.

The code is designed to run completely locally (no external APIs), assuming you have the dataset and Ollama installed.

---

## 1. Repository structure

Expected layout:

```text
net-rag/
├─ README.md                        # This file
├─ requirements.txt                 # Minimal dependencies to run the project
├─ requirements_frozen.txt          # Full pip-freeze snapshot (for exact reproduction)

├─ data/
│  ├─ raw/
│  │  └─ UNSW-NB15/
│  │     ├─ UNSW_NB15_training-set.csv
│  │     └─ UNSW_NB15_testing-set.csv
│  └─ processed/
│     └─ chroma_unsw_open_set/     # Chroma index (auto-created on first run)

└─ scripts/
   ├─ unsw_open_set_rag_demo.py     # Core open-set + RAG + LLM demo (CLI)
   └─ streamlit_unsw_open_set_app.py# Streamlit UI wrapper around the same pipeline
```

If the `data/processed/chroma_unsw_open_set` folder does not exist, it will be created automatically the first time the demo builds the Chroma index.

---

## 2. Python dependencies

This repo uses two dependency files:

1. **`requirements.txt`** – minimal dependencies needed to run the project 

   ```text
   numpy==1.26.4
   pandas==2.3.3
   scikit-learn==1.7.2

   streamlit==1.51.0

   chromadb==1.3.5

   langchain==0.2.17
   langchain-core==0.2.43
   langchain-community==0.2.19
   langchain-ollama==1.0.0

   ollama==0.6.1
   ```

2. **`requirements_frozen.txt`** – full `pip freeze` snapshot from the author’s local virtual environment.
   - This is only for **exact reproduction** of the author’s environment.
   - It may be harder to install on a different machine.

The recommended installation is:

```bash
pip install -r requirements.txt
```

---

## 3. Setup instructions

### 4.1. Clone / unpack the repository

If you received this as a zip:

```bash
unzip net-rag.zip
cd net-rag
```

Otherwise:

```bash
git clone <this-repo-url>
cd net-rag
```

### 3.2. Create and activate a virtual environment (Python 3.11)

From the project root (`net-rag/`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 3.3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer the exact original environment, you can instead do:

```bash
pip install -r requirements_frozen.txt
```

…but this is not recommended unless strictly necessary.

---

## 5. Dataset: UNSW‑NB15

You need the **UNSW‑NB15** dataset CSVs:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Place them exactly here:

```text
net-rag/
└─ data/
   └─ raw/
      └─ UNSW-NB15/
         ├─ UNSW_NB15_training-set.csv
         └─ UNSW_NB15_testing-set.csv
```

The scripts assume these exact filenames and paths.

---

## 5. Ollama & local models

This project uses **Ollama** for both:

- Embeddings (`nomic-embed-text`)
- LLM (`llama3.1`)

### 5.1. Install Ollama (if not already installed)

See the official instructions:

```text
https://ollama.com/download
```

After installation, ensure the Ollama service is running (usually automatic on macOS).

### 5.2. Pull required models

From a terminal:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

The code expects Ollama to be available at:

- `http://localhost:11434`

and uses:

- `OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")`
- `ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.0)`

---

## 6. Running the core demo (CLI)

From the project root with the virtual environment **activated**:

```bash
python scripts/unsw_open_set_rag_demo.py
```

What this script does:

1. Loads the UNSW‑NB15 training and testing CSVs.
2. Holds out the **Shellcode** family as unknown at train time (`UNKNOWN_FAMILIES = {"Shellcode"}`).
3. Chooses numeric feature columns and splits train/validation/test (known vs unknown families).
4. Trains:
   - A **binary Random Forest** on benign vs malicious.
   - A **family‑level Random Forest + calibrated probabilities** on malicious flows from known families.
5. Runs:
   - Leakage sanity check (train on shuffled labels).
   - Closed‑set and open‑set metrics for the binary RF.
   - Family‑level open‑set evaluation.
6. Builds or loads a **Chroma** index in `data/processed/chroma_unsw_open_set/`:
   - Indexes up to `MAX_INDEX_FLOWS` flows from `df_train_known`.
   - Stores text descriptions and metadata (`family`, `label`, `row_idx`).
7. Runs three **curated demo flows**:
   - One `KNOWN_BENIGN`
   - One `KNOWN_MALICIOUS`
   - One `UNKNOWN_MALICIOUS` (held‑out `Shellcode` family)
8. For each demo flow:
   - Retrieves k nearest neighbors from Chroma.
   - Applies the open‑set rule:
     - `KNOWN_BENIGN`, `KNOWN_MALICIOUS`, or `UNKNOWN_MALICIOUS`
     - Includes a “rescue” and “veto” mechanism using RAG neighbors.
   - Calls the LLM to:
     - Explain the flow in plain language.
     - Confirm / adjust the label (subject to safety policies).
     - Recommend an action: `DO_NOTHING`, `BLOCK_IP`, `RATE_LIMIT_PORT`, or `MONITOR_HOST`.
   - Prints accuracy of the open‑set heuristic and the final LLM+policy decisions.

All output is printed to the terminal for inspection.

---

## 7. Running the Streamlit app

To launch the interactive UI:

```bash
streamlit run scripts/streamlit_unsw_open_set_app.py
```

Streamlit will print a local URL, typically:

```text
http://localhost:8501
```

Open that URL in a browser.

### What you can do in the UI

- Select flows either:
  - **Randomly** by ground‑truth open‑set label (`KNOWN_BENIGN`, `KNOWN_MALICIOUS`, `UNKNOWN_MALICIOUS`), optionally focusing on misclassified cases; or
  - **By index**, using either:
    - Position in `df_test_known` / `df_test_unknown` (`iloc`), or
    - Original UNSW row index (`df.index`).
- For each selected flow, the UI shows:
  1. **Flow summary**
     - Dataset (`test_known` or `test_unknown`)
     - UNSW row index
     - Ground‑truth open‑set label
     - Ground‑truth attack family and binary label
     - Serialized flow description (no labels).
  2. **Model outputs**
     - Binary RF: `p_benign`, `p_malicious`, RF confidence.
     - Family RF: predicted family, confidence, and `family_unknown` flag.
     - Open‑set label (model heuristic) and the **derived family suggestion**, plus textual reasons.
  3. **RAG neighbor summary**
     - `sim_max`, `sim_mean`
     - `mal_frac` (fraction of malicious neighbors)
     - `family_diversity`
     - Table of top‑k nearest neighbors: family, label, similarity, `row_idx`.
  4. **LLM section**
     - Final decision (after policy)
     - Recommended action
     - Raw LLM output (including post‑policy lines).

This UI is an interactive front‑end over the same core logic used in `unsw_open_set_rag_demo.py`.

---

## 8. Configuration notes

A few important constants are defined at the top of `scripts/unsw_open_set_rag_demo.py`:

- `UNKNOWN_FAMILIES = {"Shellcode"}`  
  Families held out at train time and treated as **unknown** during evaluation.

- `OPEN_SET_CONF_THRESH = 0.60`  
  Base threshold on RF confidence for simple open‑set binary checks.

- `FAMILY_CONF_THRESH = 0.80`  
  Confidence threshold for family RF to decide if a family prediction is “known” vs “unknown”.

- `MAX_INDEX_FLOWS = 10_000`  
  Maximum number of flows to index in Chroma (for speed).

- `REBUILD_CHROMA`  
  - If `True`, rebuilds the Chroma index from scratch.
  - If `False`, reuses an existing index in `data/processed/chroma_unsw_open_set`.

These can be changed if needed, but the submitted version is configured to run out‑of‑the‑box with the described dataset and dependencies.

---

## 9. Reproducibility summary

To reproduce the main results and UI:

1. Use **Python 3.11**.
2. Create a virtual environment and run:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the UNSW‑NB15 CSVs in `data/raw/UNSW-NB15/` with the exact filenames.
4. Install Ollama and pull `nomic-embed-text` and `llama3.1` models.
5. Run:

   ```bash
   python scripts/unsw_open_set_rag_demo.py
   ```

6. Optionally launch the UI:

   ```bash
   streamlit run scripts/streamlit_unsw_open_set_app.py
   ```

That is all that is needed to run the project on a fresh machine.