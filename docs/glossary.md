Glossary
========

1. Active elements: a part of data structure where computation needs to be done. It can be a graph node, a graph edge or a subgraph.
2. Topology-driven algorithm: an algorithm whose active elements are all the elements in the involved data structures. Bellman-Ford algorithm for solving single-source shortest path (SSSP) problem is an example: all edges in the input graph are active in rounds.
3. Data-driven algorithm: an algorithm whose active elements depends on previously processed work items. Dijkstra's SSSP algorithm is an example: active nodes in the priority queue are neighbors of previously processed nodes.
4. Pull-style algorithm: an algorithm whose operator reads from the neighborhood of an active element and updates the active element itself.
5. Push-style algorithm: an algorithm whose operator reads from the active element itself and updates the neighborhood of the active element.
