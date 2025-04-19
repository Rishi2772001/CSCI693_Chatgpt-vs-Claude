def convert(s, numRows):
    """
    Convert a string to a zigzag pattern and return the result read line by line.
    
    Args:
        s: The input string to be converted
        numRows: Number of rows in the zigzag pattern
        
    Returns:
        str: The converted string read line by line
    """
    # Handle edge cases
    if numRows == 1 or numRows >= len(s):
        return s
    
    # Initialize rows
    rows = [''] * numRows
    
    # Variables to track current direction and row
    index = 0
    step = 1  # 1 means going down, -1 means going up
    
    # Process each character in the string
    for char in s:
        # Add current character to the appropriate row
        rows[index] += char
        
        # Change direction if we hit the first or last row
        if index == 0:
            step = 1  # Start moving down
        elif index == numRows - 1:
            step = -1  # Start moving up
        
        # Move to the next row
        index += step
    
    # Combine all rows to get the final result
    return ''.join(rows)