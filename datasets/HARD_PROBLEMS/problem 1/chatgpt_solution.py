def isMatch(s: str, p: str) -> bool:
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        # If we've reached the end of the pattern,
        # the string must also be fully matched.
        if j == len(p):
            return i == len(s)
        
        # Check if the first character of the remaining string matches the current pattern character.
        first_match = i < len(s) and p[j] in {s[i], '.'}
        
        # If there's a '*' as the next character in the pattern,
        # it means we can either skip the '*' and its preceding element,
        # or use the '*' to match one occurrence (if there's a match).
        if j + 1 < len(p) and p[j + 1] == '*':
            return dp(i, j + 2) or (first_match and dp(i + 1, j))
        else:
            # Otherwise, if the current characters match, continue with the next characters.
            return first_match and dp(i + 1, j + 1)
    
    return dp(0, 0)