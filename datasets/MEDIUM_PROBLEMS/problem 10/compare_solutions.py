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
import sys

# Increase recursion limit for deep chain graphs.
sys.setrecursionlimit(10**6)

# Module settings for Eventual Safe States
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

##############################################
# Standard Testing & Performance Measurement
##############################################
def run_test_and_measure(func, graph, expected):
    """
    Runs a single test: func(graph)
    Measures wall-clock time, CPU time, and peak memory.
    Returns a dictionary with the result, correctness, error, and performance metrics.
    """
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()
    
    try:
        result = func(graph)
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
    
    # For eventual safe nodes, sort the result and expected for fair comparison.
    correct = (sorted(result) == sorted(expected))
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

def test_solution(solution_func):
    results = []
    for graph, expected in TEST_CASES:
        metrics = run_test_and_measure(solution_func, graph, expected)
        results.append(metrics)
    return results

##############################################
# Learning Curve Generation for Eventual Safe States
##############################################
def generate_chain_graph(n):
    """
    Generates a simple chain graph with n nodes:
      For nodes 0 to n-2, node i points to node i+1;
      node n-1 has no outgoing edges.
    This graph is acyclic and every node is safe.
    """
    graph = []
    for i in range(n):
        if i < n - 1:
            graph.append([i + 1])
        else:
            graph.append([])
    return graph

def generate_learning_curve_complexity_judge(solution_func, sizes, num_runs=3):
    """
    For each town size in 'sizes', generates a chain graph using generate_chain_graph(n),
    runs solution_func(graph) num_runs times to obtain the average execution time,
    and performs log–log regression to estimate the complexity exponent.
    Returns (sizes, avg_times, slope).
    """
    avg_times = []
    for n in sizes:
        run_times = []
        graph = generate_chain_graph(n)
        for _ in range(num_runs):
            t_start = time.perf_counter()
            solution_func(graph)
            t_end = time.perf_counter()
            run_times.append(t_end - t_start)
        avg_times.append(sum(run_times) / len(run_times))
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve.png"):
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("Town size (n)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for eventualSafeNodes")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

##############################################
# Main Comparison for Eventual Safe States
##############################################
def main():
    # Import solution modules and get the eventualSafeNodes functions.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod  = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "eventualSafeNodes") or not hasattr(claude_mod, "eventualSafeNodes"):
        raise ImportError("Both modules must define an eventualSafeNodes(graph) function.")
    
    chatgpt_func = chatgpt_mod.eventualSafeNodes
    claude_func  = claude_mod.eventualSafeNodes

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    def test_chatgpt_coverage():
        for graph, expected in TEST_CASES:
            chatgpt_func(graph)

    def test_claude_coverage():
        for graph, expected in TEST_CASES:
            claude_func(graph)

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
        graph, expected = testcase
        print(f"Test #{i}: n={len(graph)}, expected={expected}")
        print("  ChatGPT => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
            r=cg_metrics["result"],
            c=cg_metrics["correct"],
            e=cg_metrics["error"],
            t=cg_metrics["elapsed_time"],
            cpu=cg_metrics["cpu_time"],
            m=cg_metrics["peak_memory"] / 1024
        ))
        print("  Claude  => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
            r=cl_metrics["result"],
            c=cl_metrics["correct"],
            e=cl_metrics["error"],
            t=cl_metrics["elapsed_time"],
            cpu=cl_metrics["cpu_time"],
            m=cl_metrics["peak_memory"] / 1024
        ))
        print()

    # Aggregate final test metrics.
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
    print(f"   LoC:             {c_loc['total']} total lines (of which {c_loc['code']} code)")
    print()
    print(f"Claude  => All Correct: {claude_all_correct}")
    print(f"   Avg Wall Time:   {claude_avg_time*1000:.3f} ms")
    print(f"   Avg CPU Time:    {claude_avg_cpu*1000:.3f} ms")
    print(f"   CPU Utilization: {claude_util:.1f}%")
    print(f"   Peak Mem (avg):  {claude_avg_mem/1024:.2f} KB")
    print(f"   Coverage:        {claude_coverage:.2f}%")
    print(f"   LoC:             {d_loc['total']} total lines (of which {d_loc['code']} code)")
    
    # Initial decision based on correctness and performance
    if not chatgpt_all_correct and claude_all_correct:
        decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        decision = "Both solutions failed at least one test. No clear winner."
    else:
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
    
    # If both solutions pass all tests, incorporate learning curve scaling (complexity exponent).
    if chatgpt_all_correct and claude_all_correct:
        sizes = [1000, 5000, 10000, 15000, 20000, 25000]
        num_runs = 3
        sizes_chat_judge, times_chat_judge, exp_chat_judge = generate_learning_curve_complexity_judge(chatgpt_func, sizes, num_runs)
        sizes_claude_judge, times_claude_judge, exp_claude_judge = generate_learning_curve_complexity_judge(claude_func, sizes, num_runs)
        print(f"\nEstimated complexity exponent (ChatGPT, eventualSafeNodes): {exp_chat_judge:.2f}")
        print(f"Estimated complexity exponent (Claude, eventualSafeNodes): {exp_claude_judge:.2f}")
        # Update decision based on scaling if the difference in exponents is at least 0.1.
        if abs(exp_chat_judge - exp_claude_judge) >= 0.1:
            if exp_chat_judge < exp_claude_judge:
                exponent_decision = "Based on scaling, ChatGPT scales better for large inputs."
            else:
                exponent_decision = "Based on scaling, Claude scales better for large inputs."
        else:
            exponent_decision = "Both solutions have similar scaling behavior."
        decision += " " + exponent_decision
    else:
        exponent_decision = "N/A"
    
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
        ["LoC (Total)", str(c_loc['total']), str(d_loc['total'])],
        ["LoC (Code)", str(c_loc['code']), str(d_loc['code'])],
        ["Exponent (ChatGPT)", f"{exp_chat_judge:.2f}" if chatgpt_all_correct and claude_all_correct else "N/A", ""],
        ["Exponent (Claude)", "", f"{exp_claude_judge:.2f}" if chatgpt_all_correct and claude_all_correct else "N/A"],
        ["Exponent-based Decision", exponent_decision, exponent_decision],
        ["Overall Decision", decision, decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    # (No code related to isInterleave is included below.)
    
if __name__ == "__main__":
    main()
