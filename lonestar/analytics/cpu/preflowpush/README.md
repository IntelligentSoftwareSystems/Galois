Preflow Push algorithm
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

This program computes the maximum flow from a given source to a given sink 
in a given directed graph using the preflow-push algorithm (also called 
push-relabel algorithm):

A. Goldberg. Efficient Graph Algorithms for Sequential and Parallel Computers. 
PhD thesis. Dept. of EECS, MIT. 1987.

It also incorporates global relabel and gap detection heuristics:

B. Cherkassy, A. Goldberg. On implementing the push-relabel method for the 
maximum flow problem. Algorithmica. 1997

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analysis/cpu/preflowpush; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./preflowpush-cpu <path-to-graph> <source-ID> <sink-ID>`
-`$ ./preflowpush-cpu <path-to-graph> <source-ID> <sink-ID> -t=20`

PERFORMANCE
--------------------------------------------------------------------------------

* In our experience, the deterministic algorithms perform much slower than the 
  non-deterministic one.

* The performance of all algorithms depend on an optimal choice of the compile 
  time constant, CHUNK_SIZE, the granularity of stolen work when work stealing is 
  enabled (via galois::steal()). The optimal value of the constant might depend on 
  the architecture, so you might want to evaluate the performance over a range of 
  values (say [16-4096]).
