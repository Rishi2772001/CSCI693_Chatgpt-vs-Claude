# testcases.py

TEST_CASES = [
    # (operations, arguments, expected results)
    {
        "operations": ["MyQueue", "push", "push", "peek", "pop", "empty"],
        "arguments": [[], [1], [2], [], [], []],
        "expected": [None, None, None, 1, 1, False]
    },
    {
        # Additional test case:
        # Create queue -> push 10 -> push 20 -> pop() returns 10 -> push 30 ->
        # peek() returns 20 -> pop() returns 20 -> pop() returns 30 -> empty() returns True.
        "operations": ["MyQueue", "push", "push", "pop", "push", "peek", "pop", "pop", "empty"],
        "arguments": [[], [10], [20], [], [30], [], [], [], []],
        "expected": [None, None, None, 10, None, 20, 20, 30, True]
    }
]
