# testcases.py

"""
This module defines a list of test cases for the Regular Expression Matching problem.

The problem:
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*'
where:
  '.' Matches any single character.
  '*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
"""

TEST_CASES = [
    # (s,   p,     expected)
    ("aa",  "a",    False),   # "a" does not match the entire string "aa"
    ("aa",  "a*",   True),    # '*' means zero or more of the preceding element 'a', so "a*" can match "aa"
    ("ab",  ".*",   True),    # ".*" means "zero or more of any character", which can match "ab"
    ("ab",  ".",    False),   # Single '.' matches exactly one char, but s has length 2
    ("aab", "c*a*b", True),   # 'c*' can match zero 'c's, 'a*' can match "aa", and 'b' matches 'b'
    ("ab",  ".*c",  False),   # Although ".*" matches "ab", there's an extra 'c' that doesn't match
    ("aaa", "ab*a*c*a", True),# This can match "aaa" under the right interpretation
    ("bbbba", ".*a*a", True), # ".*" can match "bbbb", 'a*' can match "", and then the final 'a'
    ("",    ".*",   True),    # ".*" can match an empty string
    ("",    "a*",   True),    # "a*" can also match an empty string
]
