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
## 3 · Interpreting the output

Running `compare_solutions.py` prints a detailed console report **and** writes a
`final_comparison.csv` containing one row per model plus a “decision” row.  
Here’s what each column (and the associated plot) means:

| Column / Asset          | What it tells you                                                            |
|-------------------------|------------------------------------------------------------------------------|
| `model`                 | Which LLM’s code is being evaluated (`chatgpt` or `claude`).                 |
| `all_tests_passed`      | `1` if *every* unit test passed, else `0`.                                   |
| `avg_wall_time_ms`      | Mean wall-clock time per test case (milliseconds).                           |
| `avg_cpu_time_ms`       | Mean CPU time per test case (milliseconds).                                  |
| `cpu_utilization_pct`   | `(CPU time / wall time) × 100`; >100 % indicates multi-core parallelism.     |
| `peak_mem_kb`           | Average of per-test peak resident-set sizes (kilobytes, via `tracemalloc`).  |
| `coverage_pct`          | Percentage of lines executed at least once (`coverage.py`).                  |
| `loc_code` / `loc_doc` / `loc_empty` / `loc_total` | Source-line breakdown: pure code, doc/comments, blank, and total. |
| `complexity_exponent`   | Slope of the fitted time-complexity curve (from log-log regression).         |
| `verdict` *(decision row only)* | Short natural-language summary (e.g., “Claude is faster”).            |
| **`Time_Complexity_Curve.png`** | PNG plot saved alongside the CSV that visualises runtime vs. input size. |

> **Tip** — Open the CSV in a spreadsheet or run the optional aggregator script
> to compare models across *all* problems in one view.

With these metrics you can see not just *whether* each solution is correct, but also
**how fast, memory-efficient, compact, and thoroughly-tested** it is—and even get an
empirical hint at its Big-O behaviour.

## 4 · Adding new problems

1. **Create a folder** under `datasets/<DIFFICULTY>/problem N/`.  
   *Example: `datasets/MEDIUM_PROBLEMS/problem 23/`*

2. **Drop in (or copy from `datasets/TEMPLATE/`) _all four_ of these files:**

   | File | Purpose |
   |------|---------|
   | `prompt.txt` | The natural-language task description shown to the LLMs. |
   | `chatgpt_solution.py` | Raw code saved directly from ChatGPT. |
   | `claude_solution.py`  | Raw code saved directly from Claude. |
   | `testcases.py`        | Define `TEST_CASES = [...]` as a list of *(input, expected_output)* tuples **and** include any helper functions you need for verification. |

3. **Add a copy of `compare_solutions.py`.**  
   Every problem folder must contain its *own* `compare_solutions.py` (the benchmarking harness).  
   The easiest way: just copy the template version and keep it unchanged unless you need extra metrics.

4. **Run the benchmark once**:

   ```bash
   cd datasets/<DIFFICULTY>/problem\ N
   python compare_solutions.py
   ```
## 5 · Contributing

Pull requests and GitHub issues are welcome 💡

* **Bugs / feature requests** – open an issue.  
* **New metrics** – submit a PR with a concise description and unit tests.  
* **More tasks** – follow *Adding new problems* above and open a PR.  

## 6 · Citation

If you use this benchmark in academic work, please cite it:

```bibtex
@misc{ganji2025chatgptvsclaude,
  author = {Rishi Ganji},
  title  = {{ChatGPT-vs-Claude}: An Open Benchmark Suite for Code Generation},
  year   = {2025},
  url    = {https://github.com/Rishi2772001/CSCI693_Chatgpt-vs-Claude}
}
```

