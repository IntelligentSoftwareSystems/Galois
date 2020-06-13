Single Source Shortest Path
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program computes the distance of shortest paths in a graph, starting from a
source node (specified by -startNode option). 

- deltaStep implements a variation on the Delta-Stepping algorithm by Meyer and
  Sanders, 2003. serDelta is its serial implementation 
- dijkstra is a serial implementation of Dijkstra's algorithm
- topo is a variation on Bellman-Ford algorithm, which visits all the nodes in the
  graph, every round, until convergence

Each algorithm has a variant that implements edge tiling, e.g. deltaTile, which
divides the edges of high-degree nodes into multiple work items for better
load balancing. 

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs having integer edge weights.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/sssp; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./sssp-cpu <path-to-graph> -algo deltaStep -delta 13 -t 40`
-`$ ./sssp-cpu <path-to-graph> -algo deltaTile -delta 13 -t 40`

PERFORMANCE  
--------------------------------------------------------------------------------

* deltaStep/deltaTile algorithms typically performs the best on high diameter
  graphs, such as road networks. Its performance is sensitive to the *delta* parameter, which is
  provided as a power-of-2 at the commandline. *delta* parameter should be tuned
  for every input graph
* topo/topoTile algorithms typically perform the best on low diameter graphs, such
  as social networks and RMAT graphs
* All algorithms rely on CHUNK_SIZE for load balancing, which needs to be
  tuned for machine and input graph. 
* Tile variants of algorithms provide better load balancing and performance
  for graphs with high-degree nodes. Tile size is controlled via
  EDGE_TILE_SIZE constant, which needs to be tuned. 
