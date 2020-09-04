Minimum Weight Spanning Tree
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program computes a minimum-weight spanning tree (MST) of an input graph.

This implementation uses a Union-Find (aka Disjoint Set) data structure to keep
track of spanning trees and to avoid cycles in the tree.  The algorithm proceeds in multiple rounds, 
where in each round, it performs two
parallel phases. One phase performs *Find* operations while the other phase
performs *Union* operations. 

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs.

- If the input is a non-symmetric graph, the program first converts it into symmetric
  graph (MST is defined for undirected/symmetric graphs only).
- If the input is a symmetric graph, the user must provide -symmetricGraph flag at
  commandline

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/spanningtree; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./minimum-spanningtree-cpu <path-to-directed-graph> -algo parallel -t 40`
-`$ ./minimum-spanningtree-cpu <path-to-symmetric-graph> -symmetricGraph -algo parallel -t 40`

PERFORMANCE  
--------------------------------------------------------------------------------

* All parallel loops in 'parallel' algorithm rely on CHUNK_SIZE parameter for load-balancing,
  which needs to be tuned for machine and input graph. 
