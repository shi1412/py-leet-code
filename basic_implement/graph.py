GRAPH = {
    'A': ['B', 'F'],
    'B': ['C', 'I', 'G'],
    'C': ['B', 'I', 'D'],
    'D': ['C', 'I', 'G', 'H', 'E'],
    'E': ['D', 'H', 'F'],
    'F': ['A', 'G', 'E'],
    'G': ['B', 'F', 'H', 'D'],
    'H': ['G', 'D', 'E'],
    'I': ['B', 'C', 'D'],
}

from collections import deque


"""
q = [A, ] -> q = []
cur = {A: ['B', 'F']}
visited = {} -> visited = {'A'}
q = [] -> q = ['B', 'F']
"""
def bfs(graph, start):
    res = []
    q = deque([start, ])
    visited = set()
    while q:
        cur = q.popleft()
        if cur not in visited:
           res.append(cur)
           visited.add(cur)
           for node in graph[cur]:
               q.append(node)
               
    return res
               
def dfs(graph, start):
    res = []
    visited = set()
    helper(res, graph, start, visited)
    return res 
    
def helper(res, graph, start, visited):
    if start not in visited:
        res.append(start)
        visited.add(start)
    
    for node in graph[start]:
        if node not in visited:
            helper(res, graph, node, visited)