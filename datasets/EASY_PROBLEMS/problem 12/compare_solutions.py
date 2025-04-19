import time
import tracemalloc
import importlib
import os
import csv
import coverage
import numpy as np
import matplotlib.pyplot as plt
import random
import string
from pygount import SourceAnalysis
from testcases import TEST_CASES

# Names of the solution modules
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

# Paths to those modules
CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

def run_test_and_measure(func, strs, expected):
    """
    Runs a single test: func(strs).
    Measures:
      - wall-clock time,
      - peak memory usage,
      - CPU time,
      - correctness (by comparing result to expected).
    Returns these metrics plus the function's actual output.
    """
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()
    
    try:
        result = func(strs)
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
    """
    Measures coverage for module_name by running test_func,
    which calls the module's solution on the test data.
    """
    cov = coverage.Coverage(source=[module_name])
    cov.start()
    test_func()
    cov.stop()
    mod = importlib.import_module(module_name)
    with open(os.devnull, 'w') as devnull:
        coverage_percent = cov.report(morfs=[mod.__file__], file=devnull)
    return coverage_percent

def measure_lines_of_code(filepath):
    """
    Uses pygount to measure lines of code in a Python file.
    """
    analysis = SourceAnalysis.from_file(filepath, "python")
    return {
        "code": analysis.code_count,
        "doc": analysis.documentation_count,
        "empty": analysis.empty_count,
        "total": analysis.code_count + analysis.documentation_count + analysis.empty_count
    }

def test_solution(solution_func):
    """
    Runs all test cases from TEST_CASES against a single solution function (longestCommonPrefix),
    collecting performance metrics for each test.
    """
    results = []
    for strs, expected in TEST_CASES:
        metrics = run_test_and_measure(solution_func, strs, expected)
        results.append(metrics)
    return results

# --------------------------
# Learning Curve Generation
# --------------------------

def generate_random_strs(length_of_each_str, num_strings=50):
    """
    Generates num_strings random lowercase strings, each of length length_of_each_str.
    """
    return [
        ''.join(random.choices(string.ascii_lowercase, k=length_of_each_str))
        for _ in range(num_strings)
    ]

def generate_learning_curve_complexity_both(chatgpt_func, claude_func, sizes, num_runs=3, num_strings=50):
    """
    Generates a learning curve for both ChatGPT and Claude solutions on the same random testcases.
    For each size in sizes, we:
      1) Generate *one* set of random strings (with length = size, number = num_strings).
      2) For each solution, run it num_runs times on that same set, and average the time.
    We then do log–log linear regression for each solution to estimate time-complexity exponent.

    Returns:
      sizes: list of input sizes (lengths),
      chat_times: average times for ChatGPT,
      chat_exp: slope from log–log regression for ChatGPT,
      claude_times: average times for Claude,
      claude_exp: slope from log–log regression for Claude.
    """
    chat_times = []
    claude_times = []
    
    for size in sizes:
        # Generate a single set of random strings for this size
        test_strs = [
            ''.join(random.choices(string.ascii_lowercase, k=size))
            for _ in range(num_strings)
        ]
        
        # Measure ChatGPT's average time
        chat_run_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            chatgpt_func(test_strs)
            t_end = time.perf_counter()
            chat_run_times.append(t_end - t_start)
        chat_times.append(sum(chat_run_times) / len(chat_run_times))
        
        # Measure Claude's average time
        claude_run_times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            claude_func(test_strs)
            t_end = time.perf_counter()
            claude_run_times.append(t_end - t_start)
        claude_times.append(sum(claude_run_times) / len(claude_run_times))
    
    # Log–log linear regression
    logsizes = np.log10(sizes)
    chat_logtimes = np.log10(chat_times)
    claude_logtimes = np.log10(claude_times)
    
    chat_slope, _ = np.polyfit(logsizes, chat_logtimes, 1)
    claude_slope, _ = np.polyfit(logsizes, claude_logtimes, 1)
    
    return sizes, chat_times, chat_slope, claude_times, claude_slope

def plot_learning_curves_both(sizes, chat_times, chat_exp, claude_times, claude_exp, filename="learning_curve.png"):
    """
    Plots both solutions' learning curves on the same log–log plot and saves to filename.
    """
    plt.figure(figsize=(8,6))
    plt.plot(sizes, chat_times, marker='o', label=f"ChatGPT (exp ≈ {chat_exp:.2f})")
    plt.plot(sizes, claude_times, marker='s', label=f"Claude (exp ≈ {claude_exp:.2f})")
    plt.xlabel("String length (L)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for longestCommonPrefix")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

def main():
    # Import the two solution modules and retrieve the solutions
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod  = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "longestCommonPrefix") or not hasattr(claude_mod, "longestCommonPrefix"):
        raise ImportError("Both modules must define a longestCommonPrefix(strs) function.")
    
    chatgpt_func = chatgpt_mod.longestCommonPrefix
    claude_func  = claude_mod.longestCommonPrefix

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    # Coverage test functions
    def test_chatgpt_coverage():
        for strs, expected in TEST_CASES:
            chatgpt_func(strs)

    def test_claude_coverage():
        for strs, expected in TEST_CASES:
            claude_func(strs)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage  = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(chatgpt_func)
    claude_results  = test_solution(claude_func)

    # Print per-test results
    for i, (testcase, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        strs, expected = testcase
        print(f"Test #{i}: input={strs!r}, expected={expected}")
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

    # Aggregate final metrics
    def avg(values):
        return sum(values) / len(values) if values else 0.0

    chatgpt_times = [r["elapsed_time"] for r in chatgpt_results]
    claude_times  = [r["elapsed_time"] for r in claude_results]
    chatgpt_cpu   = [r["cpu_time"] for r in chatgpt_results]
    claude_cpu    = [r["cpu_time"] for r in claude_results]
    chatgpt_mems  = [r["peak_memory"] for r in chatgpt_results]
    claude_mems   = [r["peak_memory"] for r in claude_results]

    chatgpt_all_correct = all(r["correct"] for r in chatgpt_results)
    claude_all_correct  = all(r["correct"] for r in claude_results)

    chatgpt_avg_time = avg(chatgpt_times)
    claude_avg_time  = avg(claude_times)
    chatgpt_avg_cpu  = avg(chatgpt_cpu)
    claude_avg_cpu   = avg(claude_cpu)
    chatgpt_avg_mem  = avg(chatgpt_mems)
    claude_avg_mem   = avg(claude_mems)

    chatgpt_util = (chatgpt_avg_cpu / chatgpt_avg_time * 100) if chatgpt_avg_time != 0 else 0
    claude_util  = (claude_avg_cpu / claude_avg_time * 100) if claude_avg_time != 0 else 0

    # Generate learning curve complexity estimation for both solutions
    sizes_for_lc = [10, 100, 1000, 5000, 10000]   # Different string lengths
    num_runs_for_lc = 3
    num_strings_for_lc = 50
    sizes_out, chat_times_lc, chat_exp, claude_times_lc, claude_exp = generate_learning_curve_complexity_both(
        chatgpt_func, claude_func, sizes_for_lc, num_runs_for_lc, num_strings_for_lc
    )

    print("==================== FINAL COMPARISON ====================")
    print(f"ChatGPT => All Correct: {chatgpt_all_correct}")
    print(f"   Avg Wall Time:   {chatgpt_avg_time*1000:.3f} ms")
    print(f"   Avg CPU Time:    {chatgpt_avg_cpu*1000:.3f} ms")
    print(f"   CPU Utilization: {chatgpt_util:.1f}%")
    print(f"   Peak Mem (avg):  {chatgpt_avg_mem/1024:.2f} KB")
    print(f"   Coverage:        {chatgpt_coverage:.2f}%")
    print(f"   LoC:             {c_loc['total']} total lines (of which {c_loc['code']} code)")
    print(f"   Estimated Complexity Exponent (ChatGPT): {chat_exp:.2f}")
    print()
    print(f"Claude  => All Correct: {claude_all_correct}")
    print(f"   Avg Wall Time:   {claude_avg_time*1000:.3f} ms")
    print(f"   Avg CPU Time:    {claude_avg_cpu*1000:.3f} ms")
    print(f"   CPU Utilization: {claude_util:.1f}%")
    print(f"   Peak Mem (avg):  {claude_avg_mem/1024:.2f} KB")
    print(f"   Coverage:        {claude_coverage:.2f}%")
    print(f"   LoC:             {d_loc['total']} total lines (of which {d_loc['code']} code)")
    print(f"   Estimated Complexity Exponent (Claude): {claude_exp:.2f}")
    print()

    # Decision logic updated to include estimated complexity exponent.
    if not chatgpt_all_correct and claude_all_correct:
        raw_decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        raw_decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        raw_decision = "Both solutions failed at least one test. No clear winner."
    else:
        # Both are correct, let's see performance:
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
                raw_decision = "Both correct & near-equal performance."
        # Incorporate estimated complexity exponent as an additional decision factor.
        if chat_exp < claude_exp - 0.1:
            exponent_decision = "ChatGPT might scale better for large inputs (lower exponent)."
        elif claude_exp < chat_exp - 0.1:
            exponent_decision = "Claude might scale better for large inputs (lower exponent)."
        else:
            exponent_decision = "Both have similar exponents; no clear winner for large inputs."
        final_decision = f"{raw_decision} Additionally, {exponent_decision}"
        raw_decision = final_decision
    decision = raw_decision

    print("\n---------------- Decision ----------------")
    print(decision)
    
    # Write final metrics to CSV
    csv_filename = os.path.join(os.path.dirname(__file__), "final_comparison.csv")
    headers = ["Metric", "ChatGPT", "Claude"]
    rows = [
        ["All Correct", str(chatgpt_all_correct), str(claude_all_correct)],
        ["Avg Wall Time (ms)", f"{chatgpt_avg_time*1000:.3f}", f"{claude_avg_time*1000:.3f}"],
        ["Avg CPU Time (ms)", f"{chatgpt_avg_cpu*1000:.3f}", f"{claude_avg_cpu*1000:.3f}"],
        ["CPU Utilization (%)", f"{chatgpt_util:.1f}", f"{claude_util:.1f}"],
        ["Peak Memory (KB)", f"{chatgpt_avg_mem/1024:.2f}", f"{claude_avg_mem/1024:.2f}"],
        ["Coverage (%)", f"{chatgpt_coverage:.2f}", f"{claude_coverage:.2f}"],
        ["LoC (Total)", str(c_loc['total']), str(d_loc['total'])],
        ["LoC (Code)", str(c_loc['code']), str(d_loc['code'])],
        ["Estimated Complexity Exponent", f"{chat_exp:.2f}", f"{claude_exp:.2f}"],
        ["Decision", decision, decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    # Plot learning curves for both solutions using the computed data
    plot_learning_curves_both(sizes_out, chat_times_lc, chat_exp, claude_times_lc, claude_exp, filename="learning_curve.png")
    print(f"Estimated complexity exponent (ChatGPT): {chat_exp:.2f}")
    print(f"Estimated complexity exponent (Claude): {claude_exp:.2f}")

if __name__ == "__main__":
    main()
