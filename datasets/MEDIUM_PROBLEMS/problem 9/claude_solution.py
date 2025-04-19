def isInterleave(s1, s2, s3):
    """
    Determine if s3 is formed by interleaving of s1 and s2.
    
    Args:
        s1: First input string
        s2: Second input string
        s3: Target string to check if it's an interleaving
        
    Returns:
        bool: True if s3 is an interleaving of s1 and s2, False otherwise
    """
    # Check if lengths match up
    if len(s1) + len(s2) != len(s3):
        return False
    
    # dp[i][j] represents if s3[0:i+j] is an interleaving of s1[0:i] and s2[0:j]
    dp = [[False for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    
    # Empty strings case
    dp[0][0] = True
    
    # Initialize first row (s1 is empty)
    for j in range(1, len(s2) + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # Initialize first column (s2 is empty)
    for i in range(1, len(s1) + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # Fill the dp table
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            # Current character of s3
            k = i + j - 1
            
            # Check if we can form s3[0:k+1] by:
            # 1. Using s1[i-1] as the last character (if it matches s3[k])
            # 2. Using s2[j-1] as the last character (if it matches s3[k])
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[k]) or (dp[i][j-1] and s2[j-1] == s3[k])
    
    return dp[len(s1)][len(s2)]