def longestCommonPrefix(strs):
    # Edge case: empty list
    if not strs:
        return ""
    
    # Start with the first string as the initial prefix
    prefix = strs[0]
    
    # Compare the current prefix with each string in the array
    for string in strs[1:]:
        # Adjust prefix length to match the current string if needed
        i = 0
        # Check character by character
        while i < len(prefix) and i < len(string) and prefix[i] == string[i]:
            i += 1
        
        # Update prefix to the common part
        prefix = prefix[:i]
        
        # If prefix becomes empty, there's no common prefix
        if not prefix:
            return ""
    
    return prefix