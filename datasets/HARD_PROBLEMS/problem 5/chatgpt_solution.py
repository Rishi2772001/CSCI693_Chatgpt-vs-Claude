def isMatch(s: str, p: str) -> bool:
    s_idx, p_idx = 0, 0
    star_idx = -1
    s_tmp_idx = -1

    while s_idx < len(s):
        # If the pattern character matches the string character or is a '?'
        if p_idx < len(p) and (p[p_idx] == s[s_idx] or p[p_idx] == '?'):
            s_idx += 1
            p_idx += 1
        # If we see a '*', record its index and the position in s
        elif p_idx < len(p) and p[p_idx] == '*':
            star_idx = p_idx
            s_tmp_idx = s_idx
            p_idx += 1
        # If there was a previous '*' in the pattern, try to use it to match one more character in s
        elif star_idx != -1:
            p_idx = star_idx + 1
            s_tmp_idx += 1
            s_idx = s_tmp_idx
        else:
            return False

    # Check for remaining '*' characters in pattern
    while p_idx < len(p) and p[p_idx] == '*':
        p_idx += 1

    return p_idx == len(p)
