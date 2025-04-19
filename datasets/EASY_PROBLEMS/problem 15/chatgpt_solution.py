def climbStairs(n: int) -> int:
    if n == 1:
        return 1
    
    # dp[i] will hold the number of ways to reach step i.
    dp = [0] * (n + 1)
    dp[0] = 1  # One way to be at step 0 (starting point).
    dp[1] = 1  # One way to reach the first step.
    
    # Each step i can be reached either from step i-1 or step i-2.
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]