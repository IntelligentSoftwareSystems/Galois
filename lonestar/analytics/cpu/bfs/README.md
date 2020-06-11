Breadth First Search
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program performs breadth-first search on an input graph, starting from a
source node (specified by -startNode option). 

Async algorithm maintains a concurrent FIFO of active nodes and uses a
for_each loop (a single parallel phase) to go over them. New active nodes are
added to the concurrent FIFO

Sync algorithm iterates over active nodes in rounds, each round, it uses a
do_all loop to iterate over currently active nodes to generate the next set of
active nodes. 

Sync2p further divides each round into two parallel do_all loops

Each algorithm has a variant that implements edge tiling, e.g. SyncTile, which
divides the edges of high-degree nodes into multiple work items for better
load balancing. 

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/bfs; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./bfs-cpu <path-to-graph> -exec PARALLEL -algo SyncTile -t 40`
-`$ ./bfs-cpu <path-to-graph> -exec SERIAL -algo SyncTile -t 40`

PERFORMANCE  
--------------------------------------------------------------------------------

* In our experience, Sync/SyncTile algorithm gives the best performance.
* Async/AsyncTile algorithm typically performs better than Sync on high diameter
  graphs, such as road networks
* All algorithms rely on CHUNK_SIZE for load balancing, which needs to be
  tuned for machine and input graph. 
* Tile variants of algorithms provide better load balancing and performance
  for graphs with high-degree nodes. Tile size is controlled via
  EDGE_TILE_SIZE constant, which needs to be tuned. 
