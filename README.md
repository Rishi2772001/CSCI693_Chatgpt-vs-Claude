# CSCI693_Chatgpt-vs-Claude

Measure—**quickly and repeatably**—how well ChatGPT and Claude solve the same programming problems.

| 🔎 Metric              | ✅ Tracked |
|------------------------|-----------|
| Unit-test correctness  | ✔ |
| Wall-clock runtime     | ✔ |
| Peak memory usage      | ✔ |
| Line coverage (%)      | ✔ |

The repo ships with a growing **dataset of coding tasks** (`datasets/`) arranged by difficulty.  
For every task you get:

* the original natural-language prompt  
* each model’s generated solution (`chatgpt_solution.py`, `claude_solution.py`)  
* a small test-suite (`testcases.py`)  
* an **out-of-the-box harness** (`compare_solutions.py`) that logs the metrics above to `final_comparison.csv`.

---

## 1 · Project goals

1. **Reproducible evaluation** – anyone can re-run every benchmark locally.  
2. **Fine-grained insight** – see where one model is faster, leaner, or more accurate.  
3. **Extensibility** – plug new problems, new models, or new metrics in minutes.

---

## 2 · Quick start

```bash
# ➊ Clone the repo
git clone https://github.com/Rishi2772001/CSCI693_Chatgpt-vs-Claude.git
cd CSCI693_Chatgpt-vs-Claude
```
### Run a single benchmark 
```bash
cd datasets/EASY_PROBLEMS/problem\ 11
python compare_solutions.py
```

