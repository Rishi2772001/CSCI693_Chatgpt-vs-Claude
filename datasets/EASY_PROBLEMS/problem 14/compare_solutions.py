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

# File/module settings
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

# Get ListNode and helper functions from chatgpt_solution if available,
# otherwise define them.
chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
if hasattr(chatgpt_mod, "ListNode"):
    ListNode = chatgpt_mod.ListNode
else:
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

def build_linked_list(lst):
    """Builds a linked list from a Python list and returns its head."""
    if not lst:
        return None
    head = ListNode(lst[0])
    curr = head
    for val in lst[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def linked_list_to_list(head):
    """Converts a linked list to a Python list."""
    result = []
    curr = head
    while curr:
        result.append(curr.val)
        curr = curr.next
    return result

def run_test_and_measure(func, lst, expected):
    """
    Runs a single test: converts lst to a linked list, calls func(head),
    converts the result back to a Python list, and measures performance.
    Returns a dictionary with the result, correctness, error, elapsed time,
    peak memory, and CPU time.
    """
    head = build_linked_list(lst)
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()
    
    try:
        new_head = func(head)
        result = linked_list_to_list(new_head)
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

def test_solution(solution_func):
    results = []
    for lst, expected in TEST_CASES:
        metrics = run_test_and_measure(solution_func, lst, expected)
        results.append(metrics)
    return results

# Generate a linked list with duplicates of size n.
def generate_input_linked_list(n):
    """
    Generates a sorted linked list with duplicates.
    For example, for n=10, it returns a list with elements:
      [0,0,1,1,2,2,...] of length n.
    """
    lst = [i // 2 for i in range(n)]
    return build_linked_list(lst)

def generate_learning_curve_complexity(solution_func, sizes, num_runs=3):
    """
    For each input size in sizes, generates a linked list (using generate_input_linked_list),
    runs solution_func(head) num_runs times to measure average execution time, then
    performs a log–log linear regression to estimate the time complexity exponent.
    Returns (sizes, avg_times, exponent).
    """
    avg_times = []
    for size in sizes:
        run_times = []
        for _ in range(num_runs):
            head = generate_input_linked_list(size)
            t_start = time.perf_counter()
            solution_func(head)
            t_end = time.perf_counter()
            run_times.append(t_end - t_start)
        avg_times.append(sum(run_times) / len(run_times))
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, intercept = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve.png"):
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("Input size (n)")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for deleteDuplicates")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

def main():
    # Import solution modules and get the deleteDuplicates functions.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod  = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "deleteDuplicates") or not hasattr(claude_mod, "deleteDuplicates"):
        raise ImportError("Both modules must define a deleteDuplicates(head) function.")
    
    chatgpt_func = chatgpt_mod.deleteDuplicates
    claude_func  = claude_mod.deleteDuplicates

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    def test_chatgpt_coverage():
        for lst, expected in TEST_CASES:
            chatgpt_func(build_linked_list(lst))

    def test_claude_coverage():
        for lst, expected in TEST_CASES:
            claude_func(build_linked_list(lst))

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
        lst, expected = testcase
        print(f"Test #{i}: input={lst}, expected={expected}")
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

    # Aggregate final metrics.
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

    # Compute learning curve complexity for both solutions
    sizes_for_lc = [10, 100, 1000, 5000, 10000]
    num_runs_for_lc = 3
    sizes_chat, times_chat, exp_chat = generate_learning_curve_complexity(chatgpt_func, sizes_for_lc, num_runs_for_lc)
    sizes_claude, times_claude, exp_claude = generate_learning_curve_complexity(claude_func, sizes_for_lc, num_runs_for_lc)

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
    
    # Plot learning curves for both solutions.
    plot_learning_curves(sizes_chat, times_chat, exp_chat, sizes_claude, times_claude, exp_claude, filename="learning_curve.png")
    print(f"Estimated complexity exponent (ChatGPT): {exp_chat:.2f}")
    print(f"Estimated complexity exponent (Claude): {exp_claude:.2f}")

if __name__ == "__main__":
    main()
