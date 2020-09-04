Minumum Spanning Tree
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark computes a minimum spanning tree in a graph. This program uses worklists for better performance.
The algorithm is implemented by successive edge-relaxations of the minimum weight edges. However, since an explicit edge-relaxation involves modifying the graph, the implementation performs edge-relaxation indirectly. This is done by keeping track of the set of nodes that have been merged, called components, which avoids modifications to the graph. Each component's size grows in each iteration, while the number of components reduces (due to components getting merged). 

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric Galois .gr graphs.
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/spanningtree; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./maximal-independent-gpu -o=<output-file> <input-graph> -symmetricGraph`
-`$ ./maximal-independent-gpu -o=rmat15.out rmat15.sgr -symmetricGraph`
