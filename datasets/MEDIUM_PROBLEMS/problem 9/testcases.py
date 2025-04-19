# testcases.py

TEST_CASES = [
    # (s1, s2, s3, expected_output)
    ("aabcc", "dbbca", "aadbbcbcac", True),   # Example 1: valid interleaving.
    ("aabcc", "dbbca", "aadbbbaccc", False),   # Example 2: invalid interleaving.
    ("", "", "", True),                         # Example 3: all strings empty.
    ("abc", "def", "abcdef", True),             # Simple concatenation.
    ("abc", "def", "adbcef", True),             # Valid interleaving: a d b c e f.
    ("a", "b", "ab", True),                     # Simple interleaving.
    ("a", "b", "ba", True),                     # Both orders are valid since each string has one char.
    ("a", "", "a", True),                       # One string is empty.
    ("", "b", "b", True),                       # One string is empty.
    ("abc", "def", "abedcf", False),            # Invalid: order of s2 not preserved (e appears before d).
]
