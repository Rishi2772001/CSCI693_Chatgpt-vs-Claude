def isMatch(s, p):
    """
    Determine if string s matches pattern p with support for wildcards.
    
    Args:
        s: Input string, containing only lowercase English letters
        p: Pattern string, containing lowercase letters, '?' and '*'
        
    Returns:
        bool: True if the pattern matches the entire string, False otherwise
    """
    # Get lengths of string and pattern
    m, n = len(s), len(p)
    
    # Create a DP table of size (m+1) x (n+1)
    # dp[i][j] represents if s[0...i-1] matches p[0...j-1]
    dp = [[False for _ in range(n+1)] for _ in range(m+1)]
    
    # Empty pattern matches empty string
    dp[0][0] = True
    
    # Handle patterns with '*' at the beginning
    for j in range(1, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                # '*' can match empty sequence (dp[i][j-1]) or multiple characters (dp[i-1][j])
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                # '?' matches any single character, or characters match exactly
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]