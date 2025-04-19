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

# Increase recursion limit (if needed)
sys.setrecursionlimit(10**6)

# Module settings for Twitter.getNewsFeed
CHATGPT_MODULE_NAME = "chatgpt_solution"
CLAUDE_MODULE_NAME  = "claude_solution"

CHATGPT_PATH = os.path.join(os.path.dirname(__file__), "chatgpt_solution.py")
CLAUDE_PATH  = os.path.join(os.path.dirname(__file__), "claude_solution.py")

##############################################
# Standard Testing & Performance Measurement
##############################################
def run_test_and_measure(ops, args, expected, TwitterClass):
    """
    Executes a sequence of operations for a Twitter instance.
    'ops' is a list of operations; 'args' is a list of argument lists.
    The first operation must be "Twitter" (to instantiate).
    Returns a dictionary with outputs, correctness, error, and performance metrics.
    """
    start_cpu_time = time.process_time()
    tracemalloc.start()
    start_wall_time = time.perf_counter()

    results = []
    instance = None
    try:
        for op, arg in zip(ops, args):
            if op == "Twitter":
                instance = TwitterClass()
                results.append(None)
            else:
                method = getattr(instance, op)
                res = method(*arg)
                results.append(res)
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

    correct = (results == expected)
    return {
        "result": results,
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
        return cov.report(morfs=[mod.__file__], file=devnull)

def measure_lines_of_code(filepath):
    analysis = SourceAnalysis.from_file(filepath, "python")
    return {
        "code": analysis.code_count,
        "doc": analysis.documentation_count,
        "empty": analysis.empty_count,
        "total": analysis.code_count + analysis.documentation_count + analysis.empty_count
    }

def test_solution(TwitterClass):
    results = []
    for test_case in TEST_CASES:
        ops = test_case["operations"]
        args = test_case["arguments"]
        expected = test_case["expected"]
        metrics = run_test_and_measure(ops, args, expected, TwitterClass)
        results.append(metrics)
    return results

##############################################
# Learning Curve Generation for Twitter.getNewsFeed
##############################################
def measure_getNewsFeed_time(TwitterClass, tweets_per_user, num_users=100, num_runs=3):
    total_time = 0.0
    for _ in range(num_runs):
        instance = TwitterClass()
        # User 1 follows all other users.
        for u in range(2, num_users + 1):
            instance.follow(1, u)
        tweet_id = 1
        # Each user posts 'tweets_per_user' tweets.
        for u in range(1, num_users + 1):
            for _ in range(tweets_per_user):
                instance.postTweet(u, tweet_id)
                tweet_id += 1
        t_start = time.perf_counter()
        instance.getNewsFeed(1)
        t_end = time.perf_counter()
        total_time += (t_end - t_start)
    return total_time / num_runs

def generate_learning_curve_complexity_twitter(TwitterClass, sizes, num_users=100, num_runs=3):
    avg_times = []
    for m in sizes:
        avg_time = measure_getNewsFeed_time(TwitterClass, m, num_users, num_runs)
        avg_times.append(avg_time)
    logsizes = np.log10(sizes)
    logtimes = np.log10(avg_times)
    slope, _ = np.polyfit(logsizes, logtimes, 1)
    return sizes, avg_times, slope

def plot_learning_curves_twitter(sizes1, times1, exp1, sizes2, times2, exp2, filename="learning_curve_twitter.png"):
    plt.figure(figsize=(8,6))
    plt.plot(sizes1, times1, marker='o', label=f"ChatGPT (exp ≈ {exp1:.2f})")
    plt.plot(sizes2, times2, marker='s', label=f"Claude (exp ≈ {exp2:.2f})")
    plt.xlabel("Tweets per user")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for Twitter.getNewsFeed")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve for Twitter.getNewsFeed saved to {filename}")

##############################################
# Main Comparison for Twitter.getNewsFeed
##############################################
def main():
    # Import Twitter classes from both solutions.
    chatgpt_mod = importlib.import_module(CHATGPT_MODULE_NAME)
    claude_mod = importlib.import_module(CLAUDE_MODULE_NAME)
    
    if not hasattr(chatgpt_mod, "Twitter") or not hasattr(claude_mod, "Twitter"):
        raise ImportError("Both modules must define a Twitter class.")
    
    Twitter_ChatGPT = chatgpt_mod.Twitter
    Twitter_Claude = claude_mod.Twitter

    print("=== Measuring lines of code (LoC) ===")
    c_loc = measure_lines_of_code(CHATGPT_PATH)
    d_loc = measure_lines_of_code(CLAUDE_PATH)
    print(f"ChatGPT solution LoC => code: {c_loc['code']}, doc/comments: {c_loc['doc']}, empty: {c_loc['empty']}, total: {c_loc['total']}")
    print(f"Claude solution LoC  => code: {d_loc['code']}, doc/comments: {d_loc['doc']}, empty: {d_loc['empty']}, total: {d_loc['total']}")
    print()

    def test_chatgpt_coverage():
        for test_case in TEST_CASES:
            ops = test_case["operations"]
            args = test_case["arguments"]
            instance = Twitter_ChatGPT()
            for op, arg in zip(ops, args):
                if op == "Twitter":
                    continue
                getattr(instance, op)(*arg)

    def test_claude_coverage():
        for test_case in TEST_CASES:
            ops = test_case["operations"]
            args = test_case["arguments"]
            instance = Twitter_Claude()
            for op, arg in zip(ops, args):
                if op == "Twitter":
                    continue
                getattr(instance, op)(*arg)

    print("=== Measuring coverage ===")
    chatgpt_coverage = measure_coverage(CHATGPT_MODULE_NAME, test_chatgpt_coverage)
    claude_coverage = measure_coverage(CLAUDE_MODULE_NAME, test_claude_coverage)
    print(f"ChatGPT solution coverage: {chatgpt_coverage:.2f}%")
    print(f"Claude solution coverage:  {claude_coverage:.2f}%")
    print()

    print("=== Running tests and measuring performance ===")
    chatgpt_results = test_solution(Twitter_ChatGPT)
    claude_results = test_solution(Twitter_Claude)

    for i, (test_case, cg_metrics, cl_metrics) in enumerate(zip(TEST_CASES, chatgpt_results, claude_results), start=1):
        ops = test_case["operations"]
        expected = test_case["expected"]
        print(f"Test #{i}: {ops}, expected={expected}")
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
    
    # Initial decision based on correctness first.
    if not chatgpt_all_correct and claude_all_correct:
        decision = "Claude's solution is better (correctness)."
    elif chatgpt_all_correct and not claude_all_correct:
        decision = "ChatGPT's solution is better (correctness)."
    elif not chatgpt_all_correct and not claude_all_correct:
        decision = "One or both solutions failed some test cases. No clear winner."
    else:
        # Both passed; decide based on wall time (and then memory if needed)
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
    
    print("\n---------------- Decision ----------------")
    print(decision)
    
    # Save partial decision and metrics to CSV.
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
        ["Decision", decision, decision]
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    
    print(f"\nFinal comparison and decision have been saved to {csv_filename}")
    
    ##############################################
    # Learning Curve Generation for Twitter.getNewsFeed
    ##############################################
    sizes = [10, 50, 100, 200, 500]  # tweets per user
    num_runs = 3
    num_users = 100

    sizes_chat_twitter, times_chat_twitter, exp_chat_twitter = generate_learning_curve_complexity_twitter(Twitter_ChatGPT, sizes, num_users, num_runs)
    sizes_claude_twitter, times_claude_twitter, exp_claude_twitter = generate_learning_curve_complexity_twitter(Twitter_Claude, sizes, num_users, num_runs)

    plt.figure(figsize=(8,6))
    plt.plot(sizes_chat_twitter, times_chat_twitter, marker='o', label=f"ChatGPT (exp ≈ {exp_chat_twitter:.2f})")
    plt.plot(sizes_claude_twitter, times_claude_twitter, marker='s', label=f"Claude (exp ≈ {exp_claude_twitter:.2f})")
    plt.xlabel("Tweets per user")
    plt.ylabel("Average execution time (s)")
    plt.title("Learning Curve Comparison for Twitter.getNewsFeed")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    learning_curve_twitter_file = os.path.join(os.path.dirname(__file__), "learning_curve_twitter.png")
    plt.savefig(learning_curve_twitter_file)
    plt.close()
    print(f"Learning curve for Twitter.getNewsFeed saved to {learning_curve_twitter_file}")
    print(f"Estimated complexity exponent (ChatGPT, Twitter.getNewsFeed): {exp_chat_twitter:.2f}")
    print(f"Estimated complexity exponent (Claude, Twitter.getNewsFeed): {exp_claude_twitter:.2f}")

    # Exponent-based decision for scaling (using a threshold of 0.05)
    if abs(exp_chat_twitter - exp_claude_twitter) >= 0.05:
        if exp_chat_twitter < exp_claude_twitter:
            exponent_decision = "Based on scaling, ChatGPT may perform better for large inputs."
        else:
            exponent_decision = "Based on scaling, Claude may perform better for large inputs."
    else:
        exponent_decision = "Both solutions have similar scaling behavior."

    print("\nExponent-based Decision:")
    print(exponent_decision)
    
    # Append exponent info and exponent-based decision to CSV.
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Exponent (Twitter.getNewsFeed)", f"{exp_chat_twitter:.2f}", f"{exp_claude_twitter:.2f}"])
        csvwriter.writerow(["Exponent-based Decision", exponent_decision, exponent_decision])
    
    print(f"\nExponent information appended to {csv_filename}")

if __name__ == "__main__":
    main()
