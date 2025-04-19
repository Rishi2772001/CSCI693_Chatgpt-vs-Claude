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
import sys
from pygount import SourceAnalysis
from testcases import TEST_CASES

# Increase recursion limit if needed
sys.setrecursionlimit(10**6)

# Module settings for Interleaving String Check
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

##############################################
# Standard Testing & Performance Measurement
##############################################
def run_test_and_measure(func, s1, s2, s3, expected):
    """
    Runs a single test: func(s1, s2, s3)
    Measures wall-clock time, CPU time, and peak memory.
    Returns a dictionary with the result, correctness, error, and performance metrics.
    """
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()
    
    try:
        result = func(s1, s2, s3)
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
    for s1, s2, s3, expected in TEST_CASES:
        metrics = run_test_and_measure(func, s1, s2, s3, expected)
        results.append(metrics)
    return results

##############################################
# Learning Curve Generation for Interleaving String Check
##############################################
def generate_interleaved_strings(L):
    """
    Generates two random strings s1 and s2 (each of length L) and produces s3 by interleaving them.
    """
    s1 = ''.join(random.choices(string.ascii_lowercase, k=L))
    s2 = ''.join(random.choices(string.ascii_lowercase, k=L))
    i, j = 0, 0
    s3_chars = []
    while i < L or j < L:
        if i < L and j < L:
            if random.choice([True, False]):
                s3_chars.append(s1[i])
                i += 1
            else:
                s3_chars.append(s2[j])
                j += 1
        elif i < L:
            s3_chars.append(s1[i])
            i += 1
        else:
            s3_chars.append(s2[j])
            j += 1
    s3 = ''.join(s3_chars)
    return s1, s2, s3

def generate_learning_curve_complexity_interleaving(solution_func, sizes, num_runs=3):
    """
    For each string length in sizes, generates random s1, s2 and a valid interleaving s3 using generate_interleaved_strings.
    Runs solution_func(s1, s2, s3) num_runs times to compute the average execution time.
    Performs log–log regression to estimate the complexity exponent.
    Returns (sizes, avg_times, exponent).
    """
    avg_times = []
    for L in sizes:
        run_times = []
        # Generate one fixed input set for fairness.
        s1, s2, s3 = generate_interleaved_strings(L)
        for _ in range(num_runs):
            t_start = time.perf_counter()
            solution_func(s1, s2, s3)
            t_end = time.perf_counter()
            run_times.append(t_end - t_start)
        avg_times.append(sum(run_times) / len(run_times))
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve_interleaving.png"):
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("String length (L)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for isInterleave")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

##############################################
# Main Comparison for Interleaving String Check
##############################################
def main():
    # Import solution modules and get the isInterleave functions.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "isInterleave") or not hasattr(claude_mod, "isInterleave"):
        raise ImportError("Both modules must define an isInterleave(s1, s2, s3) function.")
    
    chatgpt_func = chatgpt_mod.isInterleave
    claude_func = claude_mod.isInterleave

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    def test_chatgpt_coverage():
        for s1, s2, s3, expected in TEST_CASES:
            chatgpt_func(s1, s2, s3)

    def test_claude_coverage():
        for s1, s2, s3, expected in TEST_CASES:
            claude_func(s1, s2, s3)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(chatgpt_func)
    claude_results = test_solution(claude_func)

    for i, (testcase, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        s1, s2, s3, expected = testcase
        print(f"Test #{i}: s1={s1!r}, s2={s2!r}, s3={s3!r}, expected={expected}")
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

    # Aggregate final test metrics.
    chatgpt_all_correct = all(r["correct"] for r in chatgpt_results)
    claude_all_correct = all(r["correct"] for r in claude_results)

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    chatgpt_times = [r["elapsed_time"] for r in chatgpt_results]
    claude_times = [r["elapsed_time"] for r in claude_results]
    chatgpt_cpu = [r["cpu_time"] for r in chatgpt_results]
    claude_cpu = [r["cpu_time"] for r in claude_results]
    chatgpt_mems = [r["peak_memory"] for r in chatgpt_results]
    claude_mems = [r["peak_memory"] for r in claude_results]

    chatgpt_avg_time = avg(chatgpt_times)
    claude_avg_time = avg(claude_times)
    chatgpt_avg_cpu = avg(chatgpt_cpu)
    claude_avg_cpu = avg(claude_cpu)
    chatgpt_avg_mem = avg(chatgpt_mems)
    claude_avg_mem = avg(claude_mems)

    chatgpt_util = (chatgpt_avg_cpu / chatgpt_avg_time * 100) if chatgpt_avg_time != 0 else 0
    claude_util = (claude_avg_cpu / claude_avg_time * 100) if claude_avg_time != 0 else 0

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
    
    # First, decide based on correctness.
    if not chatgpt_all_correct and claude_all_correct:
        decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        decision = "Both solutions failed at least one test. No clear winner."
    else:
        # Both solutions are correct. Now, consider performance.
        if chatgpt_avg_time < claude_avg_time:
            decision = "Both correct; ChatGPT is faster (less wall time)."
        elif claude_avg_time < chatgpt_avg_time:
            decision = "Both correct; Claude is faster (less wall time)."
        else:
            if chatgpt_avg_mem < claude_avg_mem:
                decision = "Both correct & same speed; ChatGPT uses less memory."
            elif claude_avg_mem < chatgpt_avg_mem:
                decision = "Both correct & same speed; Claude uses less memory."
            else:
                decision = "Both correct & near-equal performance. It's a tie!"
    
    # Now, include the scaling behavior based on the learning curve exponents.
    sizes = [50, 100, 200, 500, 1000]
    num_runs = 3
    sizes_chat, times_chat, exp_chat = generate_learning_curve_complexity_interleaving(chatgpt_func, sizes, num_runs)
    sizes_claude, times_claude, exp_claude = generate_learning_curve_complexity_interleaving(claude_func, sizes, num_runs)
    
    print(f"\nEstimated complexity exponent (ChatGPT, isInterleave): {exp_chat:.2f}")
    print(f"Estimated complexity exponent (Claude, isInterleave): {exp_claude:.2f}")
    
    # Update decision by incorporating the scaling exponent.
    if chatgpt_all_correct and claude_all_correct:
        if abs(exp_chat - exp_claude) >= 0.1:
            if exp_chat < exp_claude:
                exponent_decision = "Based on scaling, ChatGPT scales better for large inputs."
            else:
                exponent_decision = "Based on scaling, Claude scales better for large inputs."
        else:
            exponent_decision = "Both solutions have similar scaling behavior."
        # Append the exponent decision to the performance decision.
        decision += " " + exponent_decision

    print("\n---------------- Decision ----------------")
    print(decision)
    
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
        ["Exponent (ChatGPT)", f"{exp_chat:.2f}", ""],
        ["Exponent (Claude)", "", f"{exp_claude:.2f}"],
        ["Exponent-based Decision", exponent_decision, exponent_decision],
        ["Overall Decision", decision, decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    # Plot the learning curve for isInterleave.
    plt.figure(figsize=(8,6))
    plt.plot(sizes_chat, times_chat, marker='o', label=f"ChatGPT (exp ≈ {exp_chat:.2f})")
    plt.plot(sizes_claude, times_claude, marker='s', label=f"Claude (exp ≈ {exp_claude:.2f})")
    plt.xlabel("String length (L)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for isInterleave")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    learning_curve_file = os.path.join(os.path.dirname(__file__), "learning_curve_interleaving.png")
    plt.savefig(learning_curve_file)
    plt.close()
    print(f"Learning curve for isInterleave saved to {learning_curve_file}")
    print(f"Estimated complexity exponent (ChatGPT, isInterleave): {exp_chat:.2f}")
    print(f"Estimated complexity exponent (Claude, isInterleave): {exp_claude:.2f}")

if __name__ == "__main__":
    main()
