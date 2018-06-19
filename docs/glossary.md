Glossary
========

-# Active elements: a part of data structure where computation needs to be done. For example, if graph is the data structure of interest, then an active element can be a node, an edge or a subgraph.
-# Topology-driven algorithm: an algorithm whose active elements are all the elements in the involved data structures. Bellman-Ford algorithm for solving single-source shortest path (SSSP) problem is an example: all edges in the input graph are active in rounds.
-# Data-driven algorithm: an algorithm whose active elements depend on previously processed work items. Dijkstra's SSSP algorithm is an example: active nodes in the priority queue are neighbors of previously processed nodes.
-# Pull-style algorithm: an algorithm whose operator reads from the neighborhood of an active element and updates the active element itself.
-# Push-style algorithm: an algorithm whose operator reads from the active element itself and updates the neighborhood of the active element.
