# testcases.py

TEST_CASES = [
    # (n, trust, expected_output)
    (2, [[1,2]], 2),
    (3, [[1,3],[2,3]], 3),
    (3, [[1,3],[2,3],[3,1]], -1),
    (1, [], 1),
    (4, [[1,3],[1,4],[2,3],[2,4],[4,3]], 3),
]
