from typing import List

def findRedundantDirectedConnection(edges: List[List[int]]) -> List[int]:
    n = len(edges)
    # cand1 and cand2 represent the two edges causing a node to have two parents.
    cand1, cand2 = None, None
    parent = {}
    # First pass: Check for a node with two parents.
    for u, v in edges:
        if v in parent:
            cand1 = [parent[v], v]
            cand2 = [u, v]
        else:
            parent[v] = u

    # Initialize Union-Find data structure.
    uf = list(range(n + 1))
    def find(x: int) -> int:
        if uf[x] != x:
            uf[x] = find(uf[x])
        return uf[x]
    
    def union(x: int, y: int) -> bool:
        rootx = find(x)
        rooty = find(y)
        if rootx == rooty:
            return False
        uf[rooty] = rootx
        return True

    # Second pass: For each edge, skip cand2 if it exists.
    for u, v in edges:
        # If this edge is cand2, skip it.
        if cand2 and [u, v] == cand2:
            continue
        # If union fails, there is a cycle.
        if not union(u, v):
            # If there was no two-parent problem, this edge is redundant.
            if not cand1:
                return [u, v]
            # If there was a two-parent problem, then cand1 is the answer.
            return cand1
    # If no cycle was detected when skipping cand2, then cand2 is redundant.
    return cand2