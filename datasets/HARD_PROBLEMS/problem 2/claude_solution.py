def longestValidParentheses(s: str) -> int:
    """
    Find the length of the longest valid (well-formed) parentheses substring.
    
    Args:
        s: A string containing only '(' and ')'
        
    Returns:
        The length of the longest valid parentheses substring
    """
    stack = [-1]  # Initialize with -1 as a base marker
    max_length = 0
    
    for i in range(len(s)):
        if s[i] == '(':
            # Push the index of opening parenthesis onto the stack
            stack.append(i)
        else:  # s[i] == ')'
            # Pop the last opening parenthesis
            stack.pop()
            
            if not stack:
                # If stack is empty, push current index as new base marker
                stack.append(i)
            else:
                # Calculate the length of valid substring ending at current position
                # Current index minus the index at the top of the stack
                current_length = i - stack[-1]
                max_length = max(max_length, current_length)
    
    return max_length