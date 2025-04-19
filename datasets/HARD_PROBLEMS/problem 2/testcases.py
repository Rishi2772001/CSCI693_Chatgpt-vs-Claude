# testcases.py

TEST_CASES = [
    # (input_string, expected_output)
    ("(()", 2),         # Example: longest valid is "()"
    (")()())", 4),      # Example: longest valid is "()()"
    ("", 0),            # Example: empty string has no valid substring
    ("()(()", 2),       # Valid substring "()" at beginning and "()" later (non-contiguous), so max contiguous valid = 2
    ("()(())", 6),      # Entire string is valid
    ("(()())", 6),      # Entire string is valid
    ("())(())", 4),     # Longest valid substring is "(())"
    ("((()))", 6),      # Entire string is valid
    ("(((((", 0),       # No valid pairs
    ("))))))", 0),      # No valid pairs
    ("()()", 4),        # Two adjacent valid pairs, total length 4
    (")(", 0),          # No valid substring
]
