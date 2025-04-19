def isMatch(s: str, p: str) -> bool:
    """
    Implement regular expression matching with support for '.' and '*'.
    
    Arguments:
        s: Input string to match
        p: Pattern string with regex operators
        
    Returns:
        True if the pattern matches the entire input string, False otherwise
    """
    # Create a DP table with dimensions (len(s) + 1) x (len(p) + 1)
    # dp[i][j] represents if s[0...i-1] matches p[0...j-1]
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    
    # Empty pattern matches empty string
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c* etc.
    for j in range(1, len(p) + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    # Fill the DP table
    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):
            if p[j-1] == '.' or p[j-1] == s[i-1]:
                # Current characters match
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                # Two cases for '*':
                # 1. Zero occurrence of the preceding element (ignore a*)
                dp[i][j] = dp[i][j-2]
                
                # 2. Multiple occurrences of the preceding element
                # Check if the preceding character in p matches the current character in s
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            else:
                # Characters don't match
                dp[i][j] = False
    
    return dp[len(s)][len(p)]
