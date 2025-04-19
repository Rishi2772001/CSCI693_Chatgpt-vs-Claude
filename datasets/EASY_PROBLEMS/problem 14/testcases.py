# testcases.py

TEST_CASES = [
    # Each test case is a tuple: (input_list, expected_output_list)
    ([1, 1, 2], [1, 2]),
    ([1, 1, 2, 3, 3], [1, 2, 3]),
    ([], []),
    ([1, 2, 3], [1, 2, 3]),  # Already no duplicates.
    ([1, 1, 1, 1], [1]),     # All duplicates.
]
