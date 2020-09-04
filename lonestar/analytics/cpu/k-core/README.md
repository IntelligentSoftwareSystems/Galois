K-Core Decomposition
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Finds the <b>k-core</b> in a graph. A k-core of a graph G is defined as a maxiaml
connected subgraph in which all vertices have degree at least k.

This is a parallel worklist push-style implementation. The initial worklist consists
of nodes that have degree less than k. These nodes will decrement the degree
of their neighbors, and the first time a neighbor's degree falls under the
specified k value, it will be added onto the worklist so it can decrement
its neighbors as it is considered removed from the graph.

INPUT
--------------------------------------------------------------------------------

This application takes in symmetric Galois .gr graphs.
You must specify the -symmetricGraph flag when running this benchmark.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/k-core/; make -j`

RUN
--------------------------------------------------------------------------------

To run on machine with a k value of 4, use the following:
`./k-core-cpu <symmetric-input-graph> -t=<num-threads> -kcore=4 -symmetricGraph`

PERFORMANCE
--------------------------------------------------------------------------------

Worklist chunk size (specified as a constant in the source code) may affect
performance based on the input provided to k-core.

There is preallocation of pages before the main computation begins: if the
statistics reported at the end of computation indicate that pages
were allocated during computation (i.e., MemAllocMid is less than MemAllocPost),
you may need to change how many pages are preallocated before computation.
