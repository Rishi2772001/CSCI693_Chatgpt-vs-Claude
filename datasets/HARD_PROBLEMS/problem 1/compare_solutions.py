import time
import tracemalloc
import importlib
import os
import csv
import coverage
import numpy as np
import matplotlib.pyplot as plt
from pygount import SourceAnalysis
from testcases import TEST_CASES
import random
import string
import sys

# Increase recursion limit if needed (for deep recursion in DFS, etc.)
sys.setrecursionlimit(10**6)

##############################################
# Module settings for Problem 1 (Regular Expression Matching)
##############################################
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

##############################################
# Standard Testing & Performance Measurement
##############################################
def run_test_and_measure(func, s, p, expected):
    """
    Runs a single test: func(s, p)
    Measures wall-clock time, CPU time, and peak memory.
    Returns a dictionary with the result, correctness, error, and performance metrics.
    """
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()
    
    try:
        result = func(s, p)
    except Exception as e:
        end_wall_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_cpu_time = time.process_time()
        return {
            "result": None,
            "correct": False,
            "error": str(e),
            "elapsed_time": end_wall_time - start_wall_time,
            "peak_memory": peak,
            "cpu_time": end_cpu_time - start_cpu_time
        }
    
    end_wall_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_cpu_time = time.process_time()
    
    correct = (result == expected)
    return {
        "result": result,
        "correct": correct,
        "error": None,
        "elapsed_time": end_wall_time - start_wall_time,
        "peak_memory": peak,
        "cpu_time": end_cpu_time - start_cpu_time
    }

def measure_coverage(module_name, test_func):
    cov = coverage.Coverage(source=[module_name])
    cov.start()
    test_func()
    cov.stop()
    
    mod = importlib.import_module(module_name)
    with open(os.devnull, 'w') as devnull:
        coverage_percent = cov.report(morfs=[mod.__file__], file=devnull)
    return coverage_percent

def measure_lines_of_code(filepath):
    analysis = SourceAnalysis.from_file(filepath, "python")
    return {
        "code": analysis.code_count,
        "doc": analysis.documentation_count,
        "empty": analysis.empty_count,
        "total": analysis.code_count + analysis.documentation_count + analysis.empty_count
    }

def test_solution(func):
    results = []
    for s, p, expected in TEST_CASES:
        metrics = run_test_and_measure(func, s, p, expected)
        results.append(metrics)
    return results

##############################################
# Learning Curve Generation for Regular Expression Matching
##############################################
def generate_random_string(L):
    """Generates a random string of length L using lowercase letters."""
    return ''.join(random.choices(string.ascii_lowercase, k=L))

def generate_random_pattern(s, wildcard_prob=0.2, star_prob=0.2):
    """
    Generates a pattern based on string s such that it still matches s.
    For each character in s:
      - With probability wildcard_prob, replace it with '?'.
      - With probability star_prob, insert a '*' before the character.
    Returns the resulting pattern.
    """
    pattern = []
    for ch in s:
        if random.random() < star_prob:
            pattern.append('*')
        if random.random() < wildcard_prob:
            pattern.append('?')
        else:
            pattern.append(ch)
    # Optionally, add a trailing '*' with a small probability.
    if random.random() < star_prob:
        pattern.append('*')
    return ''.join(pattern)

def generate_learning_curve_complexity(func, sizes, num_runs=3, wildcard_prob=0.2, star_prob=0.2):
    """
    For each string length L in sizes, generates a random string s and a pattern p (derived from s)
    such that the pattern still matches s. Runs func(s, p) num_runs times, computes the average
    execution time, and applies log–log regression to estimate the complexity exponent.
    Returns (sizes, avg_times, exponent).
    """
    avg_times = []
    for L in sizes:
        run_times = []
        s = generate_random_string(L)
        p = generate_random_pattern(s, wildcard_prob, star_prob)
        for _ in range(num_runs):
            t_start = time.perf_counter()
            func(s, p)
            t_end = time.perf_counter()
            run_times.append(t_end - t_start)
        avg_times.append(sum(run_times) / len(run_times))
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve_regex.png"):
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("String length (L)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for Regular Expression Matching")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

##############################################
# Main Comparison for Regular Expression Matching
##############################################
def main():
    # Import solution modules and get the isMatch function from each.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod  = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "isMatch") or not hasattr(claude_mod, "isMatch"):
        raise ImportError("Both modules must define an isMatch(s, p) function.")
    
    chatgpt_func = chatgpt_mod.isMatch
    claude_func  = claude_mod.isMatch

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    def test_chatgpt_coverage():
        for s, p, expected in TEST_CASES:
            chatgpt_func(s, p)

    def test_claude_coverage():
        for s, p, expected in TEST_CASES:
            claude_func(s, p)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage  = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(chatgpt_func)
    claude_results  = test_solution(claude_func)

    for i, (testcase, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        s, p, expected = testcase
        print(f"Test #{i}: s={s!r}, p={p!r}, expected={expected}")
        print("  ChatGPT => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
            r=cg_metrics["result"],
            c=cg_metrics["correct"],
            e=cg_metrics["error"],
            t=cg_metrics["elapsed_time"],
            cpu=cg_metrics["cpu_time"],
            m=cg_metrics["peak_memory"]/1024
        ))
        print("  Claude  => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
            r=cl_metrics["result"],
            c=cl_metrics["correct"],
            e=cl_metrics["error"],
            t=cl_metrics["elapsed_time"],
            cpu=cl_metrics["cpu_time"],
            m=cl_metrics["peak_memory"]/1024
        ))
        print()

    chatgpt_all_correct = all(r["correct"] for r in chatgpt_results)
    claude_all_correct  = all(r["correct"] for r in claude_results)

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    chatgpt_times = [r["elapsed_time"] for r in chatgpt_results]
    claude_times  = [r["elapsed_time"] for r in claude_results]
    chatgpt_cpu   = [r["cpu_time"] for r in chatgpt_results]
    claude_cpu    = [r["cpu_time"] for r in claude_results]
    chatgpt_mems  = [r["peak_memory"] for r in chatgpt_results]
    claude_mems   = [r["peak_memory"] for r in claude_results]

    chatgpt_avg_time = avg(chatgpt_times)
    claude_avg_time  = avg(claude_times)
    chatgpt_avg_cpu  = avg(chatgpt_cpu)
    claude_avg_cpu   = avg(claude_cpu)
    chatgpt_avg_mem  = avg(chatgpt_mems)
    claude_avg_mem   = avg(claude_mems)

    chatgpt_util = (chatgpt_avg_cpu / chatgpt_avg_time * 100) if chatgpt_avg_time != 0 else 0
    claude_util  = (claude_avg_cpu / claude_avg_time * 100) if claude_avg_time != 0 else 0

    print("==================== FINAL COMPARISON ====================")
    print(f"ChatGPT => All Correct: {chatgpt_all_correct}")
    print(f"   Avg Wall Time:   {chatgpt_avg_time*1000:.3f} ms")
    print(f"   Avg CPU Time:    {chatgpt_avg_cpu*1000:.3f} ms")
    print(f"   CPU Utilization: {chatgpt_util:.1f}%")
    print(f"   Peak Mem (avg):  {chatgpt_avg_mem/1024:.2f} KB")
    print(f"   Coverage:        {chatgpt_coverage:.2f}%")
    print(f"   LoC:             {measure_lines_of_code(CHATGPT_PATH)['total']} total lines (of which {measure_lines_of_code(CHATGPT_PATH)['code']} code)")
    print()
    print(f"Claude  => All Correct: {claude_all_correct}")
    print(f"   Avg Wall Time:   {claude_avg_time*1000:.3f} ms")
    print(f"   Avg CPU Time:    {claude_avg_cpu*1000:.3f} ms")
    print(f"   CPU Utilization: {claude_util:.1f}%")
    print(f"   Peak Mem (avg):  {claude_avg_mem/1024:.2f} KB")
    print(f"   Coverage:        {claude_coverage:.2f}%")
    print(f"   LoC:             {measure_lines_of_code(CLAUDE_PATH)['total']} total lines (of which {measure_lines_of_code(CLAUDE_PATH)['code']} code)")
    
    # Determine raw decision based on wall time and memory (if tests passed)
    if chatgpt_all_correct and claude_all_correct:
        if chatgpt_avg_time < claude_avg_time:
            raw_decision = "Both correct; ChatGPT is faster (less wall time)."
        elif claude_avg_time < chatgpt_avg_time:
            raw_decision = "Both correct; Claude is faster (less wall time)."
        else:
            if chatgpt_avg_mem < claude_avg_mem:
                raw_decision = "Both correct & same speed; ChatGPT uses less memory."
            elif claude_avg_mem < chatgpt_avg_mem:
                raw_decision = "Both correct & same speed; Claude uses less memory."
            else:
                raw_decision = "Both correct & near-equal performance. It's a tie!"
    else:
        raw_decision = "One or both solutions failed some test cases. No clear winner."

    # Final decision: if both passed, incorporate learning curve exponent;
    # otherwise, final decision is the raw decision.
    if chatgpt_all_correct and claude_all_correct:
        sizes = [50, 100, 200, 500, 1000]
        num_runs = 3
        sizes_chat, times_chat, exp_chat = generate_learning_curve_complexity(chatgpt_func, sizes, num_runs)
        sizes_claude, times_claude, exp_claude = generate_learning_curve_complexity(claude_func, sizes, num_runs)
        
        # Use a threshold for exponent difference (e.g., 0.1)
        if exp_chat < exp_claude - 0.1:
            exponent_decision = "ChatGPT might scale better for large inputs (lower exponent)."
        elif exp_claude < exp_chat - 0.1:
            exponent_decision = "Claude might scale better for large inputs (lower exponent)."
        else:
            exponent_decision = "Both have similar exponents; no clear winner for large inputs."
        
        final_decision = f"{raw_decision} Additionally, {exponent_decision}"
    else:
        final_decision = raw_decision
        exp_chat = "N/A"
        exp_claude = "N/A"

    print("\n---------------- Decision ----------------")
    print(final_decision)
    
    csv_filename = os.path.join(os.path.dirname(__file__), "final_comparison.csv")
    headers = ["Metric", "ChatGPT", "Claude"]
    rows = [
        ["All Correct", str(chatgpt_all_correct), str(claude_all_correct)],
        ["Avg Wall Time (ms)", f"{chatgpt_avg_time*1000:.3f}", f"{claude_avg_time*1000:.3f}"],
        ["Avg CPU Time (ms)", f"{chatgpt_avg_cpu*1000:.3f}", f"{claude_avg_cpu*1000:.3f}"],
        ["CPU Utilization (%)", f"{chatgpt_util:.1f}", f"{claude_util:.1f}"],
        ["Peak Memory (KB)", f"{chatgpt_avg_mem/1024:.2f}", f"{claude_avg_mem/1024:.2f}"],
        ["Coverage (%)", f"{chatgpt_coverage:.2f}", f"{claude_coverage:.2f}"],
        ["LoC (Total)", str(measure_lines_of_code(CHATGPT_PATH)['total']), str(measure_lines_of_code(CLAUDE_PATH)['total'])],
        ["LoC (Code)", str(measure_lines_of_code(CHATGPT_PATH)['code']), str(measure_lines_of_code(CLAUDE_PATH)['code'])],
        ["Learning Curve Exponent", f"{exp_chat:.2f}" if isinstance(exp_chat, float) else exp_chat, f"{exp_claude:.2f}" if isinstance(exp_claude, float) else exp_claude],
        ["Final Decision", final_decision, final_decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    ##############################################
    # Learning Curve Generation for Regular Expression
    ##############################################
    sizes = [50, 100, 200, 500, 1000]
    num_runs = 3
    
    sizes_chat_wc, times_chat_wc, exp_chat_wc = generate_learning_curve_complexity(chatgpt_func, sizes, num_runs)
    sizes_claude_wc, times_claude_wc, exp_claude_wc = generate_learning_curve_complexity(claude_func, sizes, num_runs)
    
    plt.figure(figsize=(8,6))
    plt.plot(sizes_chat_wc, times_chat_wc, marker='o', label=f"ChatGPT (exp ≈ {exp_chat_wc:.2f})")
    plt.plot(sizes_claude_wc, times_claude_wc, marker='s', label=f"Claude (exp ≈ {exp_claude_wc:.2f})")
    plt.xlabel("String length (L)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for Regular Expression Matching")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    learning_curve_file = os.path.join(os.path.dirname(__file__), "learning_curve_regex.png")
    plt.savefig(learning_curve_file)
    plt.close()
    print(f"Learning curve for Regular Expression Matching saved to {learning_curve_file}")
    print(f"Estimated complexity exponent (ChatGPT, isMatch): {exp_chat_wc:.2f}")
    print(f"Estimated complexity exponent (Claude, isMatch): {exp_claude_wc:.2f}")

if __name__ == "__main__":
    main()
