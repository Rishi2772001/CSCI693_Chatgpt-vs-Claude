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
import sys

# Increase recursion limit (not strictly needed for MinStack but set as a precaution)
sys.setrecursionlimit(10**6)

# Module settings for MinStack
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

##############################################
# Standard Testing & Performance Measurement
##############################################
def run_test_and_measure_solution(solution_class, operations, arguments, expected):
    """
    Executes a test case for the MinStack class.
    
    'operations' is a list of operation names (first should be "MinStack" to instantiate).
    'arguments' is a list of argument lists for each operation.
    'expected' is a list of expected outputs.
    
    Returns a dictionary with:
      - result: list of outputs.
      - correct: True if outputs match the expected list exactly.
      - error: exception message if any.
      - elapsed_time: wall-clock time in seconds.
      - peak_memory: peak memory in bytes.
      - cpu_time: CPU time in seconds.
    """
    try:
        start_cpu_time = time.process_time()
        tracemalloc.start()
        start_wall_time = time.perf_counter()
        
        results = []
        instance = None
        for op, args in zip(operations, arguments):
            if op == "MinStack":
                instance = solution_class()
                results.append(None)
            else:
                method = getattr(instance, op)
                output = method(*args)
                results.append(output)
        
        end_wall_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_cpu_time = time.process_time()
        
        correct = (results == expected)
        return {
            "result": results,
            "correct": correct,
            "error": None,
            "elapsed_time": end_wall_time - start_wall_time,
            "peak_memory": peak,
            "cpu_time": end_cpu_time - start_cpu_time
        }
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

def measure_coverage(module_name, test_func):
    """
    Measures coverage for `module_name` by running a test function that
    exercises the code from that module. Returns a coverage percentage (float).
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
    Measures lines of code in a Python file using pygount.
    Returns a dict with code, doc, empty, total counts.
    """
    analysis = SourceAnalysis.from_file(filepath, "python")
    return {
        "code": analysis.code_count,
        "doc": analysis.documentation_count,
        "empty": analysis.empty_count,
        "total": analysis.code_count + analysis.documentation_count + analysis.empty_count
    }

def test_solution(solution_class):
    """
    Runs all test cases against the given MinStack class and returns a list of
    metrics (one dict per test case).
    """
    results = []
    for test_case in TEST_CASES:
        operations = test_case["operations"]
        arguments = test_case["arguments"]
        expected = test_case["expected"]
        metrics = run_test_and_measure_solution(solution_class, operations, arguments, expected)
        results.append(metrics)
    return results

##############################################
# Learning Curve Generation for MinStack
##############################################
def simulate_minstack_operations(solution_class, num_ops):
    """
    Simulates a series of operations on a new MinStack:
      - Push num_ops random integers
      - Then call getMin() num_ops times
    Returns the total elapsed time for the getMin() calls (push time is excluded).
    """
    instance = solution_class()
    # Push random integers
    nums = [random.randint(-1000, 1000) for _ in range(num_ops)]
    for num in nums:
        instance.push(num)
    # Time how long it takes to call getMin() num_ops times
    start = time.perf_counter()
    for _ in range(num_ops):
        instance.getMin()
    end = time.perf_counter()
    return end - start

def generate_learning_curve_complexity_minstack(solution_class, sizes, num_runs=3):
    """
    For each size n in sizes, we simulate the MinStack operations using
    `simulate_minstack_operations(solution_class, n)` num_runs times,
    compute average execution time, and do a log–log fit to estimate complexity.
    Returns (sizes, avg_times, exponent).
    """
    avg_times = []
    for n in sizes:
        run_times = []
        for _ in range(num_runs):
            elapsed = simulate_minstack_operations(solution_class, n)
            run_times.append(elapsed)
        avg_times.append(sum(run_times) / len(run_times))
    
    # Perform log–log regression to estimate exponent
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves_minstack(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve_minstack.png"):
    """
    Plots the learning curves for ChatGPT and Claude solutions, each with an exponent in the label.
    """
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("Number of operations")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for MinStack")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

##############################################
# Main Comparison for MinStack
##############################################
def main():
    # Dynamically import the two solution modules
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod  = importlib.import_module(CLAUDE_MODULE_NAME)
    
    # Ensure each module defines a MinStack class
    if not hasattr(chatgpt_mod, "MinStack") or not hasattr(claude_mod, "MinStack"):
        raise ImportError("Both modules must define a MinStack class.")
    
    chatgpt_class = chatgpt_mod.MinStack
    claude_class = claude_mod.MinStack

    # Measure LoC
    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    # Coverage test functions
    def test_chatgpt_coverage():
        for test_case in TEST_CASES:
            operations = test_case["operations"]
            arguments = test_case["arguments"]
            instance = chatgpt_class()
            for op, args in zip(operations, arguments):
                if op == "MinStack":
                    continue
                getattr(instance, op)(*args)

    def test_claude_coverage():
        for test_case in TEST_CASES:
            operations = test_case["operations"]
            arguments = test_case["arguments"]
            instance = claude_class()
            for op, args in zip(operations, arguments):
                if op == "MinStack":
                    continue
                getattr(instance, op)(*args)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    # Run tests
    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(chatgpt_class)
    claude_results = test_solution(claude_class)

    # Print per-test info
    for i, (test_case, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        ops = test_case["operations"]
        expected = test_case["expected"]
        print(f"Test #{i}: operations={ops}, expected={expected}")
        print("  ChatGPT => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, "
              "cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
                r=cg_metrics["result"],
                c=cg_metrics["correct"],
                e=cg_metrics["error"],
                t=cg_metrics["elapsed_time"],
                cpu=cg_metrics["cpu_time"],
                m=cg_metrics["peak_memory"] / 1024
        ))
        print("  Claude  => output: {r}, correct: {c}, error: {e}, wall-time: {t:.6f}s, "
              "cpu-time: {cpu:.6f}s, peak-mem: {m:.2f}KB".format(
                r=cl_metrics["result"],
                c=cl_metrics["correct"],
                e=cl_metrics["error"],
                t=cl_metrics["elapsed_time"],
                cpu=cl_metrics["cpu_time"],
                m=cl_metrics["peak_memory"] / 1024
        ))
        print()

    # Check correctness across all test cases
    chatgpt_all_correct = all(r["correct"] for r in chatgpt_results)
    claude_all_correct  = all(r["correct"] for r in claude_results)

    # Helper to average times, etc.
    def avg(values):
        return sum(values) / len(values) if values else 0.0

    # Collect performance stats
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

    # Approx CPU utilization
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

    # Decide correctness first
    if not chatgpt_all_correct and claude_all_correct:
        decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        decision = "One or both solutions failed some test cases. No clear winner."
    else:
        # Both are fully correct; now compare performance
        if chatgpt_avg_time < claude_avg_time:
            decision = "Both correct; ChatGPT is faster (less wall time)."
        elif claude_avg_time < chatgpt_avg_time:
            decision = "Both correct; Claude is faster (less wall time)."
        else:
            # If times are ~ equal, compare memory
            if chatgpt_avg_mem < claude_avg_mem:
                decision = "Both correct & same speed; ChatGPT uses less memory."
            elif claude_avg_mem < chatgpt_avg_mem:
                decision = "Both correct & same speed; Claude uses less memory."
            else:
                decision = "Both correct & near-equal performance. It's a tie!"

    print("\n---------------- Decision ----------------")
    print(decision)

    # Prepare CSV rows
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
        ["Decision", decision, decision]
    ]

    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")

    # If either solution failed correctness, skip exponent-based learning curve
    if not (chatgpt_all_correct and claude_all_correct):
        print("\nOne or both solutions failed. Skipping learning curve generation for MinStack.")
        return
    
    ##############################################
    # Learning Curve Generation for MinStack
    ##############################################
    # We simulate a series of push operations + getMin calls
    sizes = [1000, 5000, 10000, 50000, 100000]
    num_runs = 3
    
    def generate_learning_curve_complexity_minstack(solution_class, sizes, num_runs=3):
        avg_times = []
        for n in sizes:
            run_times = []
            for _ in range(num_runs):
                elapsed = simulate_minstack_operations(solution_class, n)
                run_times.append(elapsed)
            avg_times.append(sum(run_times) / len(run_times))
        logsizes = np.log10(sizes)
        logtimes = np.log10(avg_times)
        slope, _ = np.polyfit(logsizes, logtimes, 1)
        return sizes, avg_times, slope

    # Generate + plot learning curves only if both solutions are correct
    sizes_chat, times_chat, exp_chat = generate_learning_curve_complexity_minstack(chatgpt_class, sizes, num_runs)
    sizes_claude, times_claude, exp_claude = generate_learning_curve_complexity_minstack(claude_class, sizes, num_runs)

    # Plot
    plot_learning_curves_minstack(
        sizes_chat, times_chat, exp_chat,
        sizes_claude, times_claude, exp_claude,
        filename="learning_curve_minstack.png"
    )
    print(f"Learning curve for MinStack operations saved to learning_curve_minstack.png")
    print(f"Estimated complexity exponent (ChatGPT, MinStack): {exp_chat:.2f}")
    print(f"Estimated complexity exponent (Claude, MinStack): {exp_claude:.2f}")

    # Print exponent-based reasoning
    exponent_decision = ""
    if abs(exp_chat - exp_claude) < 0.1:
        exponent_decision = "Both have similar scaling exponents for large n."
    elif exp_chat < exp_claude:
        exponent_decision = "ChatGPT might scale better for large n (lower exponent)."
    else:
        exponent_decision = "Claude might scale better for large n (lower exponent)."
    
    print("\nExponent-based Decision:")
    print(exponent_decision)

    # We can append exponent data + exponent-based decision to the CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Exponent (ChatGPT)", f"{exp_chat:.2f}", ""])
        csvwriter.writerow(["Exponent (Claude)", "", f"{exp_claude:.2f}"])
        csvwriter.writerow(["Exponent-based Decision", exponent_decision, exponent_decision])

if __name__ == "__main__":
    main()
