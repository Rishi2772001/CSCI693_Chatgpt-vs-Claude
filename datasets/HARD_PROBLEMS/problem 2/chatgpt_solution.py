def longestValidParentheses(s: str) -> int:
    # Stack to keep track of indices; initialize with -1 to handle edge cases.
    stack = [-1]
    max_len = 0
    
    for i, char in enumerate(s):
        if char == '(':
            # Push the index of '(' onto the stack.
            stack.append(i)
        else:
            # Pop the previous index for a matching '('.
            stack.pop()
            if not stack:
                # If the stack is empty, push the current index as a new base.
                stack.append(i)
            else:
                # Calculate the length of the current valid substring.
                max_len = max(max_len, i - stack[-1])
    
    return max_len