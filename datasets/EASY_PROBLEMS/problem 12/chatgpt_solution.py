from typing import List

def longestCommonPrefix(strs: List[str]) -> str:
    if not strs:
        return ""
    
    # Start with the first string as the initial prefix.
    prefix = strs[0] 
    
    # Compare the current prefix with each string.
    for s in strs[1:]:
        # Reduce the prefix until s starts with the prefix.
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix