from collections import deque
from typing import List

def eventualSafeNodes(graph: List[List[int]]) -> List[int]:
    """
    Find all safe nodes in a directed graph.
    
    A safe node is a node from which every path leads to a terminal node.
    A terminal node is a node with no outgoing edges.
    
    Args:
        graph: A list of lists where graph[i] contains the nodes that node i points to
        
    Returns:
        A sorted list of all safe nodes
    """
    n = len(graph)
    
    # Create a reversed graph where edges point to their source nodes
    # This will help us do a reverse topological sort
    reversed_graph = [[] for _ in range(n)]
    out_degree = [0] * n
    
    # Build the reversed graph and count outgoing edges for each node
    for i in range(n):
        out_degree[i] = len(graph[i])
        for neighbor in graph[i]:
            reversed_graph[neighbor].append(i)
    
    # Initialize queue with terminal nodes (nodes with no outgoing edges)
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
    
    # Perform a topological sort starting from terminal nodes
    safe_nodes = []
    while queue:
        node = queue.popleft()
        safe_nodes.append(node)
        
        # For each node that points to the current safe node
        for predecessor in reversed_graph[node]:
            out_degree[predecessor] -= 1
            # If all outgoing paths from predecessor now lead to safe nodes
            if out_degree[predecessor] == 0:
                queue.append(predecessor)
    
    # Sort the result
    return sorted(safe_nodes)