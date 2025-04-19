# testcases.py

TEST_CASES = [
    # (input array of strings, expected longest common prefix)
    (["flower", "flow", "flight"], "fl"),
    (["dog", "racecar", "car"], ""),
    (["interspecies", "interstellar", "interstate"], "inters"),
    (["abc"], "abc"),
    (["", "abc"], ""),
    (["cir", "car"], "c"),
    (["prefix", "preach", "prevent"], "pre"),
]
