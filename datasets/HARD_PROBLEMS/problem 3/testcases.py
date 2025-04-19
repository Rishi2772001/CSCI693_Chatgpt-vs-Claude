# testcases.py

TEST_CASES = [
    {
        # Sample test case provided in the problem statement.
        "operations": ["TextEditor", "addText", "deleteText", "addText", "cursorRight", "cursorLeft", "deleteText", "cursorLeft", "cursorRight"],
        "arguments": [[], ["leetcode"], [4], ["practice"], [3], [8], [10], [2], [6]],
        "expected": [None, None, 4, None, "etpractice", "leet", 4, "", "practi"]
    },
    {
        # Test case with various cursor moves and deletion.
        # Sequence:
        # 1. addText("hello") -> "hello|"
        # 2. cursorLeft(2) -> moves cursor left to "hel|lo", returns "hel"
        # 3. cursorRight(1) -> moves right to "hell|o", returns "hell"
        # 4. deleteText(3) -> deletes last 3 chars from left ("hell" -> "h"), returns 3
        # 5. cursorLeft(10) -> moves left as far as possible, returns ""
        # 6. cursorRight(2) -> moves right (restoring "ho"), returns "ho"
        "operations": ["TextEditor", "addText", "cursorLeft", "cursorRight", "deleteText", "cursorLeft", "cursorRight"],
        "arguments": [[], ["hello"], [2], [1], [3], [10], [2]],
        "expected": [None, None, "hel", "hell", 3, "", "ho"]
    },
    {
        # Test case with deletion exceeding the available text.
        # 1. addText("abc") -> "abc|"
        # 2. deleteText(5) -> deletes all 3 characters, returns 3
        # 3. cursorLeft(1) -> nothing to move, returns ""
        # 4. cursorRight(2) -> still nothing to move, returns ""
        "operations": ["TextEditor", "addText", "deleteText", "cursorLeft", "cursorRight"],
        "arguments": [[], ["abc"], [5], [1], [2]],
        "expected": [None, None, 3, "", ""]
    },
    {
        # Test case where the text editor remains empty.
        # 1. cursorLeft(5) -> returns "" (no text)
        # 2. cursorRight(5) -> returns "" (no text)
        # 3. deleteText(5) -> returns 0 (nothing deleted)
        "operations": ["TextEditor", "cursorLeft", "cursorRight", "deleteText"],
        "arguments": [[], [5], [5], [5]],
        "expected": [None, "", "", 0]
    }
]
