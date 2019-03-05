K-Core Decomposition
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Finds the <b>k-core</b> in a graph. A k-core is defined as a subgraph of a 
graph in which all vertices of degree less than k have been removed from the 
graph. The remaining nodes must all have a degree of at least k.

This is a parallel worklist push-style implementation. The initial worklist consists
of nodes that have degree less than k. These nodes will decrement the degree
of their neighbors, and the first time a neighbor's degree falls under the
specified k value, it will be added onto the worklist so it can decrement
its neighbors as it is considered removed from the graph.

INPUT
--------------------------------------------------------------------------------

Takes in **symmetric** Galois .gr graphs. The results obtained
from passing in non-symmetric graphs are not guaranteed to be correct nor make
sense.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/kcore/; make -j

RUN
--------------------------------------------------------------------------------

To run on machine with a k value of 4, use the following:
`./kcore <symmetric-input-graph> -t=<num-threads> -kcore=4`

PERFORMANCE
--------------------------------------------------------------------------------

Worklist chunk size (specified as a constant in the source code) may affect
performance based on the input provided to k-core.

There is preallocation of pages before the main computation begins: if the
statistics reported at the end of computation indicate that pages
were allocated during computation (i.e., MemAllocMid is less than MemAllocPost),
you may need to change how many pages are preallocated before computation.
