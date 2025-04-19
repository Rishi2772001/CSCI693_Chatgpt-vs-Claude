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

# Module settings
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

# --------------------------
# Helper functions for testing linked operations (for MyQueue)
# --------------------------
def run_test_and_measure_solution(solution_class, operations, arguments, expected):
    """
    Executes a test case for the MyQueue class.
    The test case consists of:
      - operations: list of operation names (first must be "MyQueue" to instantiate)
      - arguments: list of argument lists for each operation.
      - expected: list of expected outputs for each operation.
    Returns a dictionary with the outputs, correctness, error, and performance metrics.
    """
    try:
        start_cpu_time = time.process_time()
        tracemalloc.start()
        start_wall_time = time.perf_counter()
        
        results = []
        instance = None
        
        for op, args in zip(operations, arguments):
            if op == "MyQueue":
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

def test_solution(solution_class):
    results = []
    for test_case in TEST_CASES:
        operations = test_case["operations"]
        arguments = test_case["arguments"]
        expected = test_case["expected"]
        metrics = run_test_and_measure_solution(solution_class, operations, arguments, expected)
        results.append(metrics)
    return results

# --------------------------
# Helper functions for learning curve (for MyQueue)
# --------------------------
def generate_learning_curve_queue(solution_class, size, num_runs=3):
    """
    For a given input size, creates a new MyQueue instance,
    pushes numbers 1..size, then pops all elements.
    Repeats num_runs times and returns the average execution time.
    """
    total_time = 0.0
    for _ in range(num_runs):
        instance = solution_class()
        t_start = time.perf_counter()
        # Push numbers 1 through size.
        for i in range(1, size + 1):
            instance.push(i)
        # Pop all elements.
        while not instance.empty():
            instance.pop()
        t_end = time.perf_counter()
        total_time += (t_end - t_start)
    return total_time / num_runs

def generate_learning_curve_complexity_queue(solution_class, sizes, num_runs=3):
    """
    For each size in sizes, measure average execution time for performing n push operations followed by n pops.
    Returns sizes, avg_times, and the estimated complexity exponent (slope from log-log regression).
    """
    avg_times = []
    for size in sizes:
        avg_time = generate_learning_curve_queue(solution_class, size, num_runs)
        avg_times.append(avg_time)
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

# --------------------------
# Helper functions for coverage and LoC
# --------------------------
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

# --------------------------
# Main compare_solutions.py for MyQueue
# --------------------------
def main():
    # Import the two solution modules and get the MyQueue classes.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "MyQueue") or not hasattr(claude_mod, "MyQueue"):
        raise ImportError("Both modules must define a MyQueue class.")
    
    chatgpt_class = chatgpt_mod.MyQueue
    claude_class = claude_mod.MyQueue

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    # Coverage test functions.
    def test_chatgpt_coverage():
        for test_case in TEST_CASES:
            operations = test_case["operations"]
            arguments = test_case["arguments"]
            instance = chatgpt_class()
            for op, args in zip(operations, arguments):
                if op == "MyQueue":
                    continue
                getattr(instance, op)(*args)

    def test_claude_coverage():
        for test_case in TEST_CASES:
            operations = test_case["operations"]
            arguments = test_case["arguments"]
            instance = claude_class()
            for op, args in zip(operations, arguments):
                if op == "MyQueue":
                    continue
                getattr(instance, op)(*args)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(chatgpt_class)
    claude_results = test_solution(claude_class)

    for i, (test_case, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        operations = test_case["operations"]
        arguments = test_case["arguments"]
        expected = test_case["expected"]
        print(f"Test #{i}: operations={operations}, arguments={arguments}, expected={expected}")
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

    # Aggregate final metrics.
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

    print("==================== AGGREGATED METRICS ====================")
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
    print()

    # Generate learning curve complexity estimation for MyQueue
    sizes = [10, 100, 1000, 5000, 10000]
    num_runs = 3
    sizes_chat, times_chat, exp_chat = generate_learning_curve_complexity_queue(chatgpt_class, sizes, num_runs)
    sizes_claude, times_claude, exp_claude = generate_learning_curve_complexity_queue(claude_class, sizes, num_runs)

    # Decision logic updated to include estimated complexity exponent.
    if not chatgpt_all_correct and claude_all_correct:
        raw_decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        raw_decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        raw_decision = "Both solutions failed at least one test. No clear winner."
    else:
        # Both are correct; evaluate performance first.
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
        # Incorporate estimated complexity exponent as an additional factor.
        if exp_chat < exp_claude - 0.1:
            exponent_decision = "ChatGPT might scale better for large inputs (lower exponent)."
        elif exp_claude < exp_chat - 0.1:
            exponent_decision = "Claude might scale better for large inputs (lower exponent)."
        else:
            exponent_decision = "Both have similar exponents; no clear winner for large inputs."
        final_decision = f"{raw_decision} Additionally, {exponent_decision}"
        raw_decision = final_decision
    decision = raw_decision

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
        ["Estimated Complexity Exponent", f"{exp_chat:.2f}", f"{exp_claude:.2f}"],
        ["Decision", decision, decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    # Plot learning curves for MyQueue.
    plt.figure(figsize=(8,6))
    plt.plot(sizes_chat, times_chat, marker='o', label=f"ChatGPT (exp ≈ {exp_chat:.2f})")
    plt.plot(sizes_claude, times_claude, marker='s', label=f"Claude (exp ≈ {exp_claude:.2f})")
    plt.xlabel("Number of operations (n)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for MyQueue")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    learning_curve_file = os.path.join(os.path.dirname(__file__), "learning_curve.png")
    plt.savefig(learning_curve_file)
    plt.close()
    print(f"Learning curve saved to {learning_curve_file}")
    print(f"Estimated complexity exponent (ChatGPT): {exp_chat:.2f}")
    print(f"Estimated complexity exponent (Claude): {exp_claude:.2f}")

if __name__ == "__main__":
    main()
