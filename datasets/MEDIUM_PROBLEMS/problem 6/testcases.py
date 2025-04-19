# testcases.py

TEST_CASES = [
    # (input_string, numRows, expected_output)
    ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),  # Example 1
    ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),   # Example 2
    ("A", 1, "A"),                             # Example 3
    ("ABCD", 2, "ACBD"),                       # Zigzag for 2 rows: "A C" and "B D" => "ACBD"
    ("ABCDEFGHIJK", 3, "AEIBDFHJCGK"),         # Rows: "AEI", "BDFHJ", "CGK"
    ("", 3, ""),                               # Edge case: empty string
    ("HELLO", 1, "HELLO"),                     # Edge case: numRows = 1 (no zigzag)
]
