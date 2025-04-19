# testcases.py

TEST_CASES = [
    # (edges, expected_output)
    (
        [[1,2],[1,3],[2,3]],
        [2,3]
    ),
    (
        [[1,2],[2,3],[3,4],[4,1],[1,5]],
        [4,1]
    ),
    (
        # Two-parent scenario:
        # Node 1 receives edges from 2 and 3.
        # Expected answer: remove the first edge [2,1] (since it occurs before [3,1] in union-find cycle detection).
        [[2,1],[3,1],[4,2],[1,4]],
        [2,1]
    ),
    (
        # Cycle-only scenario (no node has two parents)
        # The extra edge [3,1] creates a cycle.
        [[1,2],[2,3],[3,1]],
        [3,1]
    ),
    (
        # Cycle-only scenario on a larger tree.
        [[1,2],[2,3],[3,4],[4,5],[5,3]],
        [5,3]
    ),
]
