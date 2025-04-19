from typing import List

def eventualSafeNodes(graph: List[List[int]]) -> List[int]:
    n = len(graph)
    # 0: unvisited, 1: visiting, 2: safe
    color = [0] * n

    def dfs(node: int) -> bool:
        # If node is already processed, return if it's safe.
        if color[node] != 0:
            return color[node] == 2

        # Mark the node as visiting.
        color[node] = 1

        for neighbor in graph[node]:
            # If any neighbor leads to a cycle (or is unsafe), node is unsafe.
            if not dfs(neighbor):
                return False

        # Mark the node as safe.
        color[node] = 2
        return True

    # Process every node.
    safe_nodes = [node for node in range(n) if dfs(node)]
    return safe_nodes
