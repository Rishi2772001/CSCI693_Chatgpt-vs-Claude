def convert(s: str, numRows: int) -> str:
    # Special cases: if only one row or the string is too short to zigzag,
    # simply return the original string.
    if numRows == 1 or numRows >= len(s):
        return s

    # Create an array for each row.
    rows = [''] * numRows
    cur_row = 0
    going_down = False

    # Traverse the string, appending characters to the appropriate row.
    for char in s:
        rows[cur_row] += char
        # Change direction if we're at the top or bottom row.
        if cur_row == 0 or cur_row == numRows - 1:
            going_down = not going_down
        cur_row += 1 if going_down else -1

    # Join all rows to form the final string.
    return ''.join(rows)