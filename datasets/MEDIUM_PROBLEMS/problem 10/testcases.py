# testcases.py

TEST_CASES = [
    # (graph, expected_output)
    (
        [[1, 2], [2, 3], [5], [0], [5], [], []],
        [2, 4, 5, 6]
    ),
    (
        [[1, 2, 3, 4], [1, 2], [3, 4], [0, 4], []],
        [4]
    ),
    (
        # Single node with no outgoing edges (terminal), so safe.
        [[]],
        [0]
    ),
    (
        # Graph with cycle: 0 -> 1, 1 -> 2, 2 -> 0, none are safe.
        [[1], [2], [0]],
        []
    ),
    (
        # Graph with self-loop (unsafe).
        [[0]],
        []
    ),
    (
        # Mixed graph.
        # 0 -> 1, 1 -> 2, 2 -> 3, 3 -> []
        # 4 -> 5, 5 -> 4 (cycle)
        # Safe nodes: 0, 1, 2, 3.
        [[1], [2], [3], []],
        [0, 1, 2, 3]
    ),
    (
        # Mixed graph where isolated safe nodes exist.
        # 0 -> [1, 2], 1 -> [], 2 -> [3], 3 -> []
        # Safe nodes: all nodes.
        [[1, 2], [], [3], []],
        [0, 1, 2, 3]
    ),
    (
        # Complex graph:
        # 0 -> 1, 2; 1 -> 2, 3; 2 -> 5; 3 -> 0, 4; 4 -> 5; 5 -> []
        # Cycle: 0 -> 1 -> 3 -> 0, so 0, 1, 3 are unsafe.
        # Safe: 2, 4, 5.
        [[1, 2], [2, 3], [5], [0, 4], [5], []],
        [2, 4, 5]
    )
]
