def findRedundantDirectedConnection(edges):
    """
    Find the redundant edge in a directed graph that started as a rooted tree.
    
    Args:
        edges: A list of directed edges [parent, child]
        
    Returns:
        The edge that can be removed to form a rooted tree
    """
    n = len(edges)
    
    # Track parent of each node (0 if no parent yet)
    parent = [0] * (n + 1)
    
    # Check if any node has two parents
    first_edge = second_edge = None
    for i, (u, v) in enumerate(edges):
        if parent[v] != 0:  # v already has a parent
            first_edge = [parent[v], v]  # Original edge to v
            second_edge = [u, v]         # New edge to v
            # Temporarily remove the second edge to check if there's still a cycle
            edges[i] = [0, 0]
            break
        parent[v] = u
    
    # Reset parent array
    parent = [0] * (n + 1)
    
    # Use union-find to detect cycle
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # Initialize each node as its own parent for union-find
    for i in range(1, n + 1):
        parent[i] = i
    
    # Check for cycles in the graph
    for u, v in edges:
        if u == 0:  # Skip the temporarily removed edge
            continue
        
        u_root = find(u)
        if u_root == v:  # Found a cycle
            # If we removed an edge earlier but still found a cycle,
            # first_edge must be causing the cycle
            if first_edge:
                return first_edge
            # Otherwise, current edge is causing the cycle
            return [u, v]
        
        # Union operation
        parent[v] = u_root
    
    # If we got here and had removed an edge, second_edge must be the answer
    # (since removing it resulted in no cycles)
    return second_edge if second_edge else edges[-1]