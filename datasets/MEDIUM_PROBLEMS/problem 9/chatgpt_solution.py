def isInterleave(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    # If total lengths don't add up, it's impossible.
    if m + n != len(s3):
        return False
    
    # dp[i][j] is True if s1[0:i] and s2[0:j] can interleave to form s3[0:i+j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Initialize first row: s1 is empty, so s2 must match s3 exactly.
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and (s2[j - 1] == s3[j - 1])

    # Initialize first column: s2 is empty, so s1 must match s3 exactly.
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] and (s1[i - 1] == s3[i - 1])

    # Fill in the rest of the table.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Two possibilities:
            # 1. The current character of s1 matches the corresponding character in s3
            #    and the remaining part is a valid interleaving.
            # 2. The current character of s2 matches the corresponding character in s3
            #    and the remaining part is a valid interleaving.
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or \
                       (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
    
    return dp[m][n]