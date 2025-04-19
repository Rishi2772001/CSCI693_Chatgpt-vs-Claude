def climbStairs(n):
    # Base cases
    if n <= 2:
        return n
    
    # Initialize first two numbers in the sequence
    first = 1  # Ways to climb 1 step
    second = 2  # Ways to climb 2 steps
    
    # Calculate ways for each step from 3 to n
    for i in range(3, n + 1):
        # Current ways = ways to reach from (n-1) + ways to reach from (n-2)
        current = first + second
        # Update values for next iteration
        first = second
        second = current
    
    return second