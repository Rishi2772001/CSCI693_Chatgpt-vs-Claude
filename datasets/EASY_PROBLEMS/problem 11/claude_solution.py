def findJudge(n, trust):
    # Initialize trust counts for each person
    # We'll use a single array where:
    # - When person A trusts person B, we decrement A's count and increment B's count
    trust_counts = [0] * (n + 1)  # +1 because people are labeled from 1 to n
    
    # Process all trust relationships
    for a, b in trust:
        trust_counts[a] -= 1  # Person a trusts someone (outgoing trust)
        trust_counts[b] += 1  # Person b is trusted (incoming trust)
    
    # Check for the judge - must have n-1 net trust (trusted by everyone else)
    # Note: We start from index 1 since people are labeled from 1 to n
    for i in range(1, n + 1):
        if trust_counts[i] == n - 1:
            return i
    
    # No judge found
    return -1