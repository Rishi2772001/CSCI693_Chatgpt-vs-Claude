# testcases.py

TEST_CASES = [
    # (input_string, pattern, expected_output)
    ("aa", "a", False),            # "a" does not match "aa"
    ("aa", "*", True),             # '*' matches any sequence
    ("cb", "?a", False),           # '?' matches 'c' but second letter mismatch
    ("adceb", "*a*b", True),       # '*' matches any sequence between letters
    ("acdcb", "a*c?b", False),      # does not match the pattern
    ("", "", True),                # both string and pattern empty
    ("", "*", True),               # '*' can match empty string
    ("abc", "a?c", True),          # '?' matches exactly one character
    ("abc", "a*d", False),         # pattern does not match
    ("abcdef", "abc*def", True),    # '*' matches an empty sequence here
    ("abcdef", "abc*ef", True),     # '*' matches 'd'
    ("abcdef", "a*e*f", True),      # '*' can match different subsequences
]
