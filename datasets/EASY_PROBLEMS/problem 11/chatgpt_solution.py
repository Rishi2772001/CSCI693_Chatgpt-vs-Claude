def findJudge(n: int, trust: list[list[int]]) -> int:
    # Initialize trust scores for each person (using 1-indexed positions)
    scores = [0] * (n + 1)
    
    # For each trust relationship [a, b]:
    # Person a trusts someone, so subtract 1.
    # Person b is trusted by someone, so add 1.
    for a, b in trust:
        scores[a] -= 1
        scores[b] += 1
    
    # The judge, if exists, will have a trust score equal to n - 1.
    # (Because everyone except themselves trusts them.)
    for person in range(1, n + 1):
        if scores[person] == n - 1:
            return person
    
    return -1