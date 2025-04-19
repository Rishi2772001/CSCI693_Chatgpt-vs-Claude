# testcases.py

TEST_CASES = [
    {
        # Sample test case provided in the problem statement.
        "operations": ["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"],
        "arguments": [[], [-2], [0], [-3], [], [], [], []],
        "expected": [None, None, None, None, -3, None, 0, -2]
    },
    {
        # Additional test case:
        # 1. Create a new MinStack.
        # 2. push(1), push(2), push(-1), push(-2)
        # 3. top() should return -2.
        # 4. getMin() should return -2.
        # 5. After pop(), getMin() should update to -1.
        # 6. After further pops, top() should return 1.
        "operations": ["MinStack", "push", "push", "push", "push", "top", "getMin", "pop", "getMin", "pop", "pop", "top"],
        "arguments": [[], [1], [2], [-1], [-2], [], [], [], [], [], [], []],
        "expected": [None, None, None, None, None, -2, -2, None, -1, None, None, 1]
    }
]
